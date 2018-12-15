import os
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from keras.models import Sequential
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json


@csrf_exempt
def index(request):

    if request.method=="POST":
        #Directory upto project
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_directory = os.path.join(BASE_DIR, 'assets')
        # scalar = MinMaxScaler()

        #Directory upto saved models
        model_json = model_directory + '\model.json'
        model_weight = model_directory + '\model.h5'

        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weight)

        ## receiving request
        received_json_data = json.loads(request.body.decode("utf-8"))
        data = received_json_data['data']
        Xnew = np.array(data)
        result = loaded_model.predict_classes(Xnew)
        class_result = []
        for item in result:
            class_result.append(str(item))
        result_proba = loaded_model.predict_proba(Xnew)
        new_result = []
        for i, (values, class_values) in enumerate(zip(result_proba, class_result)):
            final_class_label = ''
            if max(values) < 0.5:
                final_class_label = '-' + class_values
            else:
                final_class_label = class_values
            new_result.append(final_class_label)
        myString = ",".join(map(str, new_result))
        final_result = '[' + myString + ']'
        return JsonResponse({'result': new_result})
        
    else:
        return JsonResponse({'msg':'Method Not Allowed'},status=422)
    
