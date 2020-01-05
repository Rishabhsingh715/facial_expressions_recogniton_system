print('''Emotion Recognition :
1. Using speech
2. Using face

      ''')

choice = int(input("Enter your choice - "))



if choice == 1 :
        print('Wait for a while....')
        import speech_recognition as sr
        import nltk
        from nltk.corpus import stopwords
        import pandas as pd
        from nltk.stem import WordNetLemmatizer
        
        r = sr.Recognizer()

        with sr.Microphone() as source :
            print("Speak : ")
            audio = r.listen(source)
            try :
                inp = r.recognize_google(audio)
                print("You said: ",inp)
            except :
                print("Sorry, could not recognize your voice")



        stop_words = stopwords.words('english')
        stop_words.extend([',','.','-','!'])

        inp = ' '.join([word for word in inp.split() if word not in stop_words])
        inp = inp.split()
        
        lmt = WordNetLemmatizer()

        text = []

        for i in inp:
            text.append(lmt.lemmatize(i, pos='v'))
        f = open('sad.txt', 'r')
        sad = f.readlines()
        f.close()

        sad_text = []
        for i in sad :
            i = i.strip('\n')
            i = lmt.lemmatize(i, pos='v')
            sad_text.append(i)
        f = open('happy.txt', 'r')
        happy = f.readlines()
        f.close()

        happy_text = []
        for i in happy :
            i = i.strip('\n')
            i = lmt.lemmatize(i, pos='v')
            happy_text.append(i)

        f = open('anger.txt', 'r')
        anger = f.readlines()
        f.close()
        anger_text = []
        for i in anger :
            i = i.strip('\n')
            i = lmt.lemmatize(i, pos='v')
            anger_text.append(i)

        f = open('fear.txt', 'r')
        fear = f.readlines()
        f.close()
        fear_text = []
        for i in fear :
            i = i.strip('\n')
            i = lmt.lemmatize(i, pos='v')
            fear_text.append(i)

        f = open('surprise.txt', 'r')
        surprise = f.readlines()
        f.close()
        surprise_text = []
        for i in surprise :
            i = i.strip('\n')
            i = lmt.lemmatize(i, pos='v')
            surprise_text.append(i)

        happy = 0
        sad = 0
        angry = 0
        fear = 0
        surprise = 0

        for i in text :
            if i in happy_text :
                happy+=1
            elif i in sad_text :
                sad+=1
            elif i in anger_text :
                angry+=1
            elif i in fear_text :
                fear+=1
            elif i in surprise_text :
                surprised+=1

        res = max(happy, sad, angry, fear, surprise)
        emotion = ['happy', 'sad', 'angry', 'fear', 'surprise']
        print(emotion[res])
        
elif choice == 2:
        print('Wait for a while....')
        
        import numpy as np
        import cv2
        from keras.preprocessing import image
        from keras.models import model_from_json


        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        cap = cv2.VideoCapture(0)

        # use "facial_expression_model_structure.json" for pre-trained model
        model = model_from_json(open("model.json", "r").read())

        # use "facial_expression_model_weights.h5" for pre-trained weights
        model.load_weights('model.hdf5')

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

        while(True):
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x,y,w,h) in faces:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                        detected_face = cv2.resize(detected_face, (48, 48))

                        img_pixels = image.img_to_array(detected_face)
                        img_pixels = np.expand_dims(img_pixels, axis = 0)
                        img_pixels /= 255
                        
                        predictions = model.predict(img_pixels)
                        
                        max_index = np.argmax(predictions[0])
                        percentage = np.amax(predictions[0]) * 100
                        percentage = str(percentage)
                        emotion = emotions[max_index]
                        emo = emotion + " " + percentage[:5]+ '%'
                        
                        
                        
                        cv2.putText(img, emo, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        
                cv2.imshow('img',img)

                if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                        break
                
        cap.release()
        cv2.destroyAllWindows()

else :
        print("Wrong choice!!")
        print(choice)
