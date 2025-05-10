from models.glm_model import run_glm
from models.nn_model1 import run_nn_model1
from models.nn_model2 import run_nn_model2
from models.nn_model3 import run_nn_model3
from models.nn_model4 import run_nn_model4
#from models.nn_model5 import run_nn_model5

def main():
    print("Running all models and comparing deviances...\n")
    results = {}

    results["GLM"] = run_glm()
   # results["NN Model 1"] = run_nn_model1()
   #results["NN Model 2"] = run_nn_model2()
   # results["NN Model 3"] = run_nn_model3()
    results["NN Model 4"] = run_nn_model4()
   # results["NN Model 5"] = run_nn_model5()

    print("\nModel Deviance Comparison:")
    for model, dev in results.items():
        print(f"{model}: {dev:.4f}")

if __name__ == "__main__":
    main()