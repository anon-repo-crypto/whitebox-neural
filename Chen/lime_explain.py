import deep_net_mc as tn
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from lime.submodular_pick import SubmodularPick
import csv
epoch = 5
net, X, Y, X_eval, Y_eval = tn.train_speck_distinguisher(40, num_rounds=epoch, diff=(0x40, 0), group_size=2, depth=1)

np.random.seed(10)
lower_bound = 1
upper_bound = 5000


# net, X, Y, X_eval, Y_eval = tn.train_speck_distinguisher(50,num_rounds=6,depth=10);


# Create an instance of LimeTabularExplainer
explainer = LimeTabularExplainer(X,
                                 mode= "classification",
                                 feature_names = [f'feature_0_{i}' if i < int(X.shape[1]/2) else f'feature_1_{i-int(X.shape[1]/2)}' for i in range(X.shape[1])],
                                 class_names=[str(i) for i in range(2)],
                                 )

def predict_fn(data):
    prob_class_1 = net.predict(data)
    prob_class_0 = 1 - prob_class_1
    return np.hstack((prob_class_0, prob_class_1)).astype(float)
# Select an instance to explain
# instance = X_eval[0].reshape(1, -1)

# Explain the instance
# for i in range(10):
#     idx = np.random.randint(lower_bound, upper_bound + 1)
#     explanation = explainer.explain_instance(X_eval[idx], predict_fn, num_features=X_eval[idx].shape[0])

    # Visualize the explanation
    # explanation.show_in_notebook(show_table=True, show_all=False)
    # explanation.save_to_file(f"./lime_plot/6_round_sample_id{idx}_gtlabel_{Y_eval[idx]}.html")
pre = predict_fn(X_eval)
sp_obj = SubmodularPick(explainer, X_eval, predict_fn, num_features=X_eval.shape[1], num_exps_desired=200,sample_size=3000,method='sample')
explanations = sp_obj.sp_explanations
i = 1
#Writing explanation data to csv file
i = 1
#Writing explanation data to csv file
with open('lime_explanation_200_samples_6round.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Instance','Feature', 'Weight', 'Label','Prediction','Accuracy'])
    j = 0
    for exp in sp_obj.sp_explanations:
        for feature, weight in exp.as_list(label=exp.available_labels()[0]):
            writer.writerow([i,feature, weight,Y_eval[sp_obj.V[j]],np.argmax(pre[sp_obj.V[j]]), pre[sp_obj.V[j]]])
        i = i +1
        j = j+1