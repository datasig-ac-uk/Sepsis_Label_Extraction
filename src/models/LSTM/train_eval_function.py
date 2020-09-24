import torch
from definitions import *
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report,roc_curve,auc
import numpy as np
def train_model(model,train_dl,test_dl,n_epochs,patience,save_dir,loss_func,optimizer):
    model.train()
    best_loss=1e10
    for epoch in range(n_epochs):
        print(epoch)
        start = time.time()
        train_losses = []
        test_losses = []
        train_preds, test_preds,train_y,test_y = [], [],[],[]
        model.train()
        #print('training')
        for step, (x,y) in enumerate(train_dl):
            optimizer.zero_grad()
            #print('step:',step)
            prediction = model(x)
            #print(prediction)
            #print(prediction.shape,y.view(-1).shape)
            loss = loss_func(prediction, y.view(-1))
            #ßprint(loss)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            #train_y.append(y.view(-1))
            #train_preds.append(prediction)
        #print(train_preds)
        model.eval()
        #print('eval')
        for step,(x,y) in enumerate(test_dl):
            test_pred = model(x)
            #test_y.append(y.view(-1))
            #print(y.view(-1).shape)
            test_losses.append(loss_func(test_pred, y).item())
            #test_preds.append(test_pred)
        #print(Y_train,train_preds)
        print("train loss=",np.mean(train_losses),"test_loss=",np.mean(test_losses))
        print('time=',time.time() - start)
        # ealystopping, save the model if new validation loss is lower
        if best_loss > np.mean(test_losses):
           best_loss = np.mean(test_losses)
           best_t = epoch
           torch.save(model.state_dict(), save_dir)
           print('save model')
        if epoch - best_t > patience:
           print('best_test_loss=',best_loss)
           break


def eval_model(train_dl,test_dl,model):
    model.eval()
    train_preds, test_preds,train_y,test_y = [], [],[],[]
    with torch.no_grad():
        for step, (x,y) in enumerate(train_dl):
            train_y.append(y.view(-1))
            train_preds.append(torch.nn.functional.softmax(model(x), dim=1))
        for step, (x,y) in enumerate(test_dl):
            test_y.append(y.view(-1))
            test_preds.append(torch.nn.functional.softmax(model(x), dim=1))

    tfm = lambda x: torch.cat(x).cpu().detach().numpy()
    y_train_true =  tfm(train_y)
    y_train_pred = np.argmax(tfm(train_preds),axis=1)
    y_test_true =  tfm(test_y)
    y_test_pred = np.argmax(tfm(test_preds),axis=1)
    train_accuracy = accuracy_score(y_train_true,y_train_pred)
    test_accuracy = accuracy_score(y_test_true,y_test_pred)
    fpr, tpr, thresholds = roc_curve(y_test_true+1, tfm(test_preds)[:,1], pos_label=2)
    
    precision, recall,f1_score,support = precision_recall_fscore_support(y_test_true,y_test_pred, average='micro')
    print(  "train_accuracy=",train_accuracy,
        "test_accuracy=",test_accuracy,
        "test_precision=", precision,
        "test_recall=", recall,
        "test_f1_score=", f1_score,
         "test_auc_score=",auc(fpr, tpr))
    print(classification_report(y_test_true, y_test_pred,digits=4))
    print(confusion_matrix(y_test_true, y_test_pred))
    
    return train_accuracy,test_accuracy,precision,recall,f1_score,auc(fpr, tpr),y_test_true,tfm(test_preds)

def patient_level_eval(df,label,pred):
    """

    :param df: origional df
    :param label: true labels
    :param pred: predicted labels from the model
    :return: evaluation statistics at the patient level
    """
