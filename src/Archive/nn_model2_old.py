# src/nn_model2_embedding_extended.py  ---------------------------------------
"""
Neural Network #2: Extended embeddings for high-cardinality categoricals
and binned continuous features, reusing cleaned data from nn_data.
- Embeds VehBrand, Area, and binned BonusMalus
- One-hot encodes VehGas
- Includes numeric features VehPower, VehAge, DrivAge, log_Density
- Predicts annual frequency using Poisson-deviance loss
"""
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from nn_data import (
    load_clean_df,           # returns cleaned DataFrame with derived cols
    poisson_freq_loss,
    exp_val                  # exposure vector
)
from glm_baseline import mean_poisson_deviance

# 1. Load cleaned DataFrame & train/val splits ------------------------
PROJ      = Path(__file__).resolve().parents[1]
df        = load_clean_df()
idx_train = joblib.load(PROJ / "data" / "splits" / "train_idx.pkl")
idx_val   = joblib.load(PROJ / "data" / "splits" / "val_idx.pkl")
train_df  = df.loc[idx_train].copy()
val_df    = df.loc[idx_val].copy()

# 2. Bin BonusMalus into quartiles safely (drop duplicate edges) -------
quantiles = train_df['BonusMalus'].quantile([0, .25, .5, .75, 1.]).values
edges     = np.unique(quantiles)
if len(edges) > 1:
    train_df['BM_bin'] = pd.cut(train_df['BonusMalus'], bins=edges, labels=False, include_lowest=True)
    val_df  ['BM_bin'] = pd.cut(val_df   ['BonusMalus'], bins=edges, labels=False, include_lowest=True)
else:
    train_df['BM_bin'] = 0
    val_df  ['BM_bin'] = 0

# 3. Numeric inputs ----------------------------------------------------
NUM_FEATS = ['VehPower','VehAge','DrivAge','log_Density']
Xn_train  = train_df[NUM_FEATS].values.astype('float32')
Xn_val    = val_df  [NUM_FEATS].values.astype('float32')
scaler    = StandardScaler().fit(Xn_train)
Xn_train  = scaler.transform(Xn_train).astype('float32')
Xn_val    = scaler.transform(Xn_val).astype('float32')

# 4. One-hot VehGas ---------------------------------------------------
Xd_train  = pd.get_dummies(train_df['VehGas'], drop_first=True)
Xd_val    = pd.get_dummies(val_df  ['VehGas'], drop_first=True)
Xd_val    = Xd_val.reindex(columns=Xd_train.columns, fill_value=0)
Xd_train  = Xd_train.values.astype('float32')
Xd_val    = Xd_val.values.astype('float32')

# 5. Embedding indices ------------------------------------------------
train_df['VehBrand'] = pd.Categorical(train_df['VehBrand'])
val_df  ['VehBrand'] = pd.Categorical(val_df['VehBrand'], categories=train_df['VehBrand'].cat.categories)
train_df['Area']     = pd.Categorical(train_df['Area'])
val_df  ['Area']     = pd.Categorical(val_df['Area'],   categories=train_df['Area'].cat.categories)

brand_codes_train = train_df['VehBrand'].cat.codes.values.astype('int32')
brand_codes_val   = val_df  ['VehBrand'].cat.codes.values.astype('int32')
area_codes_train  = train_df['Area'].cat.codes.values.astype('int32')
area_codes_val    = val_df  ['Area'].cat.codes.values.astype('int32')
bm_codes_train    = train_df['BM_bin'].astype('int32').values
bm_codes_val      = val_df  ['BM_bin'].astype('int32').values

n_brands = train_df['VehBrand'].nunique()
n_areas  = train_df['Area'].nunique()
n_bmbins = train_df['BM_bin'].nunique()

# 6. Targets & frequency ----------------------------------------------
y_train = (train_df['ClaimNb'] / train_df['Exposure']).astype('float32').values
y_val   = (val_df  ['ClaimNb'] / val_df  ['Exposure']).astype('float32').values

# 7. Build the Wide+Deep with embeddings -----------------------------
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Concatenate,
    Dense, Dropout, BatchNormalization, Lambda
)
from tensorflow.keras.models import Model

# Inputs
inp_num   = Input(shape=(Xn_train.shape[1],), name='num_in')
inp_gas   = Input(shape=(Xd_train.shape[1],), name='gas_in')
inp_brand = Input(shape=(), dtype='int32', name='brand_in')
inp_area  = Input(shape=(), dtype='int32', name='area_in')
inp_bm    = Input(shape=(), dtype='int32', name='bmbin_in')

# Embeddings
emb_brand = Embedding(n_brands, 8, name='emb_brand')(inp_brand)
emb_brand = Flatten()(emb_brand)
emb_area  = Embedding(n_areas, 5, name='emb_area')(inp_area)
emb_area  = Flatten()(emb_area)
emb_bm    = Embedding(n_bmbins,4, name='emb_bmbin')(inp_bm)
emb_bm    = Flatten()(emb_bm)

# Deep path
x = BatchNormalization()(inp_num)
x = Concatenate()([x, emb_brand, emb_area, emb_bm])
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

# Wide path for VehGas
y = Dense(32, activation='relu')(inp_gas)

# Merge
combined = Concatenate()([x, y])

# Output with log-link
logit = Dense(1)(combined)
logit = Lambda(lambda t: tf.clip_by_value(t,-5.0,5.0))(logit)
out   = Lambda(lambda t: tf.exp(t)+1e-6)(logit)

model = Model(inputs=[inp_num, inp_gas, inp_brand, inp_area, inp_bm], outputs=out)
model.summary()

# 8. Compile ----------------------------------------------------------
opt = tf.keras.optimizers.Adam(1e-4, clipnorm=1.0, clipvalue=5.0)
model.compile(optimizer=opt, loss=poisson_freq_loss, metrics=[poisson_freq_loss])

# 9. Train ------------------------------------------------------------
history = model.fit(
    [Xn_train, Xd_train, brand_codes_train, area_codes_train, bm_codes_train],
    y_train,
    validation_data=(
        [Xn_val,   Xd_val,   brand_codes_val,   area_codes_val,   bm_codes_val],
        y_val
    ),
    epochs=100,
    batch_size=2048,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=2
)

# 10. Evaluate & save predictions ------------------------------------
freq_pred = model.predict([
    Xn_val, Xd_val, brand_codes_val, area_codes_val, bm_codes_val
], batch_size=4096).flatten()
count_pred = freq_pred * exp_val
y_val_count = (y_val * exp_val).astype('float32')
dev = mean_poisson_deviance(y_val_count, count_pred)
print(f"Val Poisson deviance: {dev:.4f}")
outf = PROJ / 'reports' / 'predictions' / 'nn2_embed_ext_val_preds.csv'
pd.DataFrame({'Obs_Count': y_val_count, 'Pred_Count': count_pred, 'Exposure': exp_val}).to_csv(outf, index=False)
print('Saved to', outf)
