from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json
from keras.models import model_from_json

def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0;
    for parameter in initializer:
        s += "parameter:"+str(parameterIndex)+"\n"
        print(parameter["dims"])
        s += "dims:"+str(parameter["dims"])+"\n"
        print(parameter["name"])
        s += "name:"+str(parameter["name"])+"\n"
        print(parameter["doubleData"])
        s += "values:"+str(parameter["doubleData"])+"\n"
        index = index + 1
        parameterIndex = index // 2
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)   
        
def export_to_custom_format(model, filename):
    with open(filename, 'w') as f:
        # Escribir el número de capas
        f.write(f'num_layers:{len(model.coefs_)}\n')

        for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
            # Escribir los coeficientes
            f.write(f'parameter:{i}\n')
            f.write(f'dims:{list(coef.shape)}\n')
            f.write('name:coefficient\n')
            f.write(f'values:{coef.flatten().tolist()}\n')

            # Escribir los intercepts
            f.write(f'parameter:{i}\n')
            f.write(f'dims:{[1, len(intercept)]}\n')
            f.write('name:intercepts\n')
            f.write(f'values:{intercept.tolist()}\n')
            
import json
from sklearn import tree

#para exportar el arbol de decisiones a json
def export_decision_tree_to_json(dtc, filename):
    data = {
        'node_count': dtc.tree_.node_count,
        'children_left': dtc.tree_.children_left.tolist(),
        'children_right': dtc.tree_.children_right.tolist(),
        'feature': dtc.tree_.feature.tolist(),
        'threshold': dtc.tree_.threshold.tolist(),
        'value': dtc.tree_.value.tolist(),
    }
    with open(filename, 'w') as file:
        json.dump(data,file)