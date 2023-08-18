#%%

import torch
from torch_models.fc_ae import FC_AE 
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device('cuda')

def train(model,optimizer,criterion,train_loader,test_loader,epochs=15):
    print(f'Starting training using device {device}...')
    for epoch in range(epochs):
        model.train()
        runloss = 0
        for inputs in tqdm(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(inputs, outputs)
            runloss += loss
            loss.backward()
            optimizer.step()

        # mses, lbls = list_loss_scores(model,criterion,test_loader)
        # plt.scatter(mses,lbls)
        # plt.show()

        print(f'epoch:{epoch}\tloss:{runloss}')

def get_anomaly_score(model,criterion,signal):
    '''
    signal.shape = [t_length,640]
    '''
    avgmse = 0
    for feature in signal:
        outputs = model(feature.unsqueeze(0))
        avgmse += criterion(outputs,feature)
    avgmse /= signal.shape[0]
    return avgmse

def list_loss_scores(model,criterion,test_set):
    '''
    Returns the average mean-square errors for the sounds in test set 
    over the central 6.4 second window of the input waves.

    PARAMETERS
    ----------
    model : torch.Module
        Pytorch autoencoder model to reconstruct input
    criterion : MSEloss
        MSEloss- no other way around it
    test_set : torch.utils.Dataset
        Test set containing log mel spectrograms
    '''
    mses = []
    lbls = []
    model.eval()
    for input,label in tqdm(test_set):
        input = input.to(device)
        avgmse = 0
        outputs = model(input)
        avgmse += criterion(outputs,input)
        avgmse /= input.shape[0]
        mses.append(avgmse)
        lbls.append(label)
    mses = [i.detach().cpu() for i in mses]
    return mses, lbls

def plot_stats(lbls,mses,pos_label):
    from sklearn.metrics import roc_curve,roc_auc_score

    anoms = [val for i,val in enumerate(mses) if lbls[i] == 0]
    norms = [val for i,val in enumerate(mses) if lbls[i] == 1]

    lbls = [not i for i in lbls if pos_label == 0]

    fpr,tpr,thresholds = roc_curve(lbls,mses)
    auc = roc_auc_score(lbls,mses)

    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(fpr,tpr,label=f'AUC = {auc.round(3)}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.subplot(1,2,2)
    plt.hist([norms,anoms],bins=100,label=['norms','anoms'])
    plt.legend()
#%%

model = FC_AE().to(device)
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    #weight_decay=1e-4
)
#%%

import anomaly_detection.toycar as tc

train_set = tc.toycar_trainset_librosa()
test_set = tc.toycar_testset_librosa()

#%%
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=8192,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=16,
    num_workers=1
)

#%%
optimizer.lr = 0.0001
train(model,optimizer,criterion,train_loader,test_loader,epochs=100)

#%%
torch.save(model.state_dict(),'saved_runs/fc_ae_8')

#%% Plot ROC
with torch.no_grad():
    mses, lbls = list_loss_scores(model,criterion,test_set)
plot_stats(lbls,mses,pos_label=0)

#%% Eval ROC

import numpy as np

indices = np.random.randint(len(test_set),size=500)
eval_set = torch.utils.data.Subset(test_set,indices)
with torch.no_grad():
    mses, lbls = list_loss_scores(model,criterion,eval_set)
plot_stats(lbls,mses,pos_label=0)

#%% Load old model

model.load_state_dict(torch.load('saved_runs/fc_ae_auc_872'))

# %% Comparing torchaudio and librosa's mel spectrograms

testfile = '/home/raimarc/lawrence-workspace/data/ToyCar/train/normal_id_01_00000000.wav'

import numpy as np
import librosa, torchaudio

y, sr = librosa.load(testfile, sr=None)
librosa_ver = librosa.feature.melspectrogram(y=y,
                                            sr=sr,
                                            n_fft=1024,
                                            hop_length=512,
                                            win_length=1024,
                                            n_mels=128,
                                            power=2,
                                            #norm='slaney',
                                            #htk=True,
                                            )

waveform, sample_rate = torchaudio.load(testfile)
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate,
                                            n_mels=128,
                                            n_fft=1024,
                                            win_length=1024,
                                            hop_length=512,
                                            power=2,
                                            mel_scale='slaney',
                                            norm = 'slaney',
                                            pad_mode = 'constant',
                                            center = True
                                            )
torch_ver = mel_transform(waveform).squeeze()

print(librosa_ver.shape)
print(torch_ver.shape)

plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
plt.imshow(librosa_ver)
plt.title('Librosa')
plt.subplot(1,2,2)
plt.imshow(torch_ver)
plt.title('Torch')
plt.show()

print(np.array([librosa_ver[0][:5],torch_ver[0][:5].numpy()]))

plt.hist([torch_ver.flatten(),librosa_ver.flatten()],bins=100)
plt.show()

# %% Getting training set mean and stdev

running_mean = 0
running_stdev = 0
for a in tqdm(train_set):
    std, mean = torch.std_mean(a)
    running_mean += mean
    running_stdev+= std
running_mean/=len(train_set)
running_stdev/=len(train_set)



# %% Quantize
torch.onnx.export(model,test_set[0][0].to(device),"fc_ae.onnx")

# %%
import tensorflow as tf
import numpy as np
from tqdm import tqdm

interpreter = tf.lite.Interpreter('saved_model/fc_ae_full_integer_quant.tflite')

output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0]
interpreter.allocate_tensors()

def get_mse(a,b,normalize=False):
    if type(a) == np.int8 or type(a) == np.int16:
        a = a.astype(float)
        b = b.astype(float)
    if normalize:
        a = (a - a.mean())/a.std()
        b = (b - b.mean())/b.std()
    return ((np.array(a)-np.array(b))**2).mean()

def tf_quantize(input_array,input_details):
    scale, zero_point = input_details['quantization']
    b = input_array/scale + zero_point
    b = b.astype(input_details['dtype'])
    return b

def tf_dequantize(input_array,input_details):
    scale, zero_point = input_details['quantization']
    b = scale*(input_array - zero_point)
    b = b.astype(np.float32)
    return b

def tf_get_anomaly_score(interpreter, in_array: np.array):
    in_array_t = in_array
    
    if input['dtype'] != np.float32:
        in_array_t = tf_quantize(in_array, input)
    
    interpreter.set_tensor(input['index'], in_array_t)
    interpreter.invoke()
    rec = interpreter.get_tensor(output['index'])

    if input['dtype'] != np.float32:
        rec = tf_dequantize(rec,output)

    mse = get_mse(rec,in_array)
    return mse

def tf_list_loss_scores(interpreter,test_set):
    mses = []
    lbls = []
    for input,label in tqdm(test_set):
        avgmse=tf_get_anomaly_score(interpreter,input.numpy())
        mses.append(avgmse)
        lbls.append(label)
    return mses, lbls

mses_tf, lbls_tf = tf_list_loss_scores(interpreter, test_set)
plot_stats(lbls_tf,mses_tf,pos_label=0)

# %%
i = np.random.randint(len(test_set))
labels = ['Anomalous','Normal']
a = test_set[i][0]
a_q = a
if input['dtype'] != np.float32:
    a_q = tf_quantize(a.numpy(),input)

interpreter.set_tensor(input['index'], a_q)
interpreter.invoke()
a_rec0 = interpreter.get_tensor(output['index'])

if input['dtype'] != np.float32:
    a_rec = tf_dequantize(a_rec0,output)

plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
plt.imshow(a_q)
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(a_rec)
plt.title('TFLite FC-AE Reconstruction')

# a_recf = a_rec.astype(float)
# a_qf = a_q.astype(float)
# a_recfn = (a_recf-a_recf.mean())/a_recf.std()
# a_qfn = (a_qf-a_qf.mean())/a_qf.std()
# print(((a_qfn-a_recfn)**2).mean())
tflite_mse = ((a-a_rec)**2).mean()
print(tflite_mse)

aa = a.to(device)
b = model(aa)
torch_mse = ((b-aa)**2).mean()
print(torch_mse)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title(f'Input idx {i}: {labels[test_set[i][1]]}')
plt.hist(a.flatten(),bins=100,label=f'Torch MSE: {torch_mse}')
plt.legend()
plt.subplot(1,2,2)
plt.title('Reconstructed')
plt.hist(a_rec.flatten(),bins=100,label=f'TFLite MSE: {tflite_mse}')
plt.legend()
print("Done")
# %%
import anomaly_detection.toycar as tc
test_set = tc.toycar_testset_librosa()

# %%

scale, zero_point = input['quantization']
plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plt.hist(a.flatten(),bins=100)
plt.title('Input Distribution')
plt.subplot(1,3,2)
plt.hist(a.flatten()/scale,bins=100)
plt.title('After Scaling')
plt.subplot(1,3,3)
plt.hist(a.flatten()/scale + zero_point,bins=100)
plt.title('After Scaling + Zero Point')

#%%

scale, zero_point = output['quantization']
a_rec1 = a_rec0 - zero_point
a_rec2 = a_rec1 * scale
# %%
