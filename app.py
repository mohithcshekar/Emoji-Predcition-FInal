#IMPORTING THE LIBRARIES
import emoji
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import string
from flask import Flask, request, render_template,redirect

 
app = Flask(__name__)

def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r',encoding='utf-8')
    gloveModel = {}
    for line in f:
        #print(line)
        splitLines = line.split()
        #print(splitlines)
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel
 
def embedding(X):
    max_len=10
    embdim=50
    #YOU HAVE TO CREATE AN EMBEDDING LAYER
    embeddinglayer=np.zeros((X.shape[0],max_len,embdim))
    #YOU HAVE CREATED AN EMBEDDING LAYER
    for i in range(X.shape[0]):
        alllines=X[i].split()
        
        for idx in range(len(alllines)):
            try:
                embeddinglayer[i][idx]=power[alllines[idx].lower()]
            except:
                embeddinglayer[i][idx]=np.zeros((50,))
    return embeddinglayer
 
 
train=pd.read_csv("train_emoji.csv")
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                    "6":":fire:",
                    "7":":face_blowing_a_kiss:",
                    "8":":chestnut:",
                    "9":":flexed_biceps:"
                   }

X_train=train['0']
Y_train=train['1']
Y_train1=to_categorical(Y_train,num_classes=5)
power=loadGloveModel("glove.6B.50d.txt")
embedding_train_matrix=embedding(X_train)
model =Sequential()
model.add(LSTM(72,input_shape=(10,50),return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(72,input_shape=(10,50)))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()
 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
 
#checkpoint=ModelCheckpoint("D:\\Desktop\\Emoji1\\best_modelf4.h5",monitor='val_acc',verbose=True,save_best_only=True)
#hist=model.fit(embedding_train_matrix,Y_train1,epochs=150,batch_size=64,shuffle=True,validation_split=0.2,callbacks=[checkpoint])
#hist=model.fit(embedding_train_matrix,Y_train1,epochs=150,batch_size=64,shuffle=True,validation_split=0.3)
 
model.load_weights("best_modelf41.h5")
lis=set(string.punctuation)

music_dictionary = {"1": "/bnqLzCsffwY,/hxcyY5ELoCQ,/YGSelD1TZWY,/7rXAxKM7j8Q",    
                    "0": "/3s5XyooFGpg,/qFkNATtc3mc,/K3nFy1wz6V0,/Z8ejNU-VCUg,/IUV1s-NfTS0,/wkay45hn-Vs",
                    "2": "/C3jlOlzSL8I,/FUpA_v_c_Vk?list=RDFUpA_v_c_Vk,/KgmeL_xuB0I",
                    "3": "/3WT2yGrWKoc,/EtGh9oC2SZ0,/abiL84EAWSY,/SSb-E-08dXQ?list=TLPQMjQwNjIwMjHHcX3P27y-Pw,/f6636xqsLGc?list=TLPQMjQwNjIwMjHHcX3P27y-Pw,/CeuQ-v43pkI?list=TLPQMjQwNjIwMjHHcX3P27y-Pw",
                    "4": "/ti2Pxrl2Nho",
                    "5":"/NujNxgy4CpA,/n9F8CEUx5m4",
                    "6":"/W-RAUYomJho,/yZFQBLUTWhA",
                    "7":"/QpKQjISfB4s,/5UfA_hGRGz0",
                    "8":"/If3ugwv1pII",
                    "9":"/EtGh9oC2SZ0",
                    "10":"/EtGh9oC2SZ0",
                    "11":"/6xAwdtATs6E"
                   }
    
    
    
listt=[]
 
 
emoji_output= ""
url_out=""
display=0
@app.route('/')
def index():
    global emoji_output,url_out,display
    #return "<h1>GFG is great platform to learn</h1>"
    return render_template("indexy.html",emoji_output=  emoji_output,  url_out= url_out,display=display)
 

@app.route('/', methods=['GET','POST'])
def index_post():
    global emoji_output,url_out, display
    display=1
    inp_text=request.form["inp_text"]

    words=[str(x.lower()) for x in inp_text.strip().split()]
    strr=[str(x) for x in words if x not in lis]
    strr=' '.join(strr)
    listt.append(strr)
    a=np.array(listt)
    listt.pop()
    powertext=embedding(a)
    powerpred=model.predict_classes(powertext)
        
    emoji_output=emoji.emojize(emoji_dictionary[str(powerpred[0])])
    
    
    number=str(powerpred[0])
    if 'mom' in words or 'mother' in words or 'mummy' in words or 'mum' in words:
        #print(music_dictionary['5'])
        lisans=music_dictionary['5']
        lisans=lisans.split(',')
        url_out=np.random.choice(lisanss)
        return redirect("/")
        
    if 'papa' in words or 'father' in words or 'daddy' in words or 'dad' in words:
        #print(music_dictionary['6'])
        lisans=music_dictionary['6']
        lisans=lisans.split(',')
        url_out=np.random.choice(lisans)
        return redirect("/")
        
    if ('friends' in words or 'friend' in words) and 'love' in words:
        #print(music_dictionary['7'])
        lisans=music_dictionary['7']
        lisans=lisans.split(',')
        url_out=np.random.choice(lisans)
        return redirect("/")
        
    if ('family' in words) and 'love' in words:
        url_out=music_dictionary['8']
        return redirect("/")
        
    if 'breakup' in words:
        url_out=music_dictionary['9']
        return redirect("/")
        
    if ('girl' in words and 'troubles' in words):
        url_out=music_dictionary['10']
        return redirect("/")
        
    if ('girl' in words and 'hates' in words):
        url_out=music_dictionary['11']
        return redirect("/")
        
    ans=music_dictionary[number]
    lines=ans.split(',')
    
    
    url_out=np.random.choice(lines)
    return redirect("/")
 
app.run(debug=True,port=5000)