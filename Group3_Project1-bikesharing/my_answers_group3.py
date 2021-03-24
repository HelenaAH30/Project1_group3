'''
Autores: Antich Homar, Helena 
         Muntaner Ferrer, Joan 
         Nordfeldt Fiol, Bo Miquel
         
Fecha: Marzo 2021

Descripción: Implementación del feedforward y backpropagation de una red neuronal
'''
import numpy as np


class NeuralNetwork(object):
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # Asignación del número de neuronas de entrada, ocultas y de salida.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Inicialización de los pesos y declaración del learning rate.
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Implementación de la función de activación para la capa oculta de la red neuronal utilizando la función
        # sigmoide (sigmoide(x) = 1 / (1 + e^-x)). Para ello utilizamos una expresión o función lambda, es decir 
        # una función anonima.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
                    

    def train(self, features, targets):
        ''' Entrenamiento de la red mediantes lotes (también conocidos como batches) de caracterísitcas (features)
            y outputs deseados (targets).
        
            Parámetros
            ---------
            
            features: array de 2 dimensiones, cada fila es un registo de datos y cada columna es una caracterísitca (o feature).
            targets: array de una dimensión con los valores de los objetivos.
        '''
        
        # Cantidad de datos.
        n_records = features.shape[0]
        
        # Variables para guardar los cambios que se deben realizar en los pesos de todas las neuronas.
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            # Feedforward: la red neuronal transforma las entradas en salidas.
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            
            # Backpropagation: la red determina la contribución del error de cada uno de los pesos que hay entre sus neuronas y guarda
            # en delta_weights_i_h o en delta_weights_h_o lo que debería cambiar cada peso para subsanar su contribución al error.
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
            
        # Cuando se tiene calculado cuanto debería variar cada peso para corregir su contribución al error se actualiza.
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implementación del feedforward utilizando datos agrupados por lotes
         
            Parámetros
            ---------
            X: lote de caracterísitcas o features
        '''
        
        # Cálculo de los valores de entrada de la capa oculta. Para ello se debe multiplicar el lote de caracterísitcas por
        # los pesos de las conexiones que hay entre los inputs (las características) y las neuronas de la capa oculta: X * W_i_h.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # np.matmul o np.dot ?
        
        # Cálculo de las salidas de las neuronas ocultas. Se utilizan los valores de entrada de cada neurona como inputs para
        # su función de activación: sigmoid(hidden_inputs) 
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        # Cálculo de los valores de entrada de la capa de salida de la red neuronal. Para ello se deben multiplicar las salidas
        # de las neuronas ocultas por los pesos que hay entre la capa de oculta y la capa de salida: hidden_outputs * W_h_o
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # np.matmul o np.dot ?
        
        # Cálculo de las salidas de las neuronas de salida de la red neuronal. Como la función de activación de las neuronas
        # de esta capa es f(x) = x, simplemente se debe igualar el output de estas neuronas a su input.
        final_outputs = final_inputs 
        
        return final_outputs, hidden_outputs


    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implementación de backpropagation
         
            Parámetros
            ---------
            final_outputs: output de toda la red neuronal
            hidden_outputs: salida de las neuronas ocultas
            X: lotes de inputs
            y: lotes de outputs deseados
            delta_weights_i_h: cambio de los pesos entre los inputs y la capa oculta
            delta_weights_h_o: cambio de los pesos entre la capa oculta y la salida
        '''
        
        # Error global de la red neuronal. Para calcularlo se debe restar el ouput predecido al output deseado: error = y - y_pred.
        error = y - final_outputs 
        
        # Cálculo de la contribución al error global de la capa de salida, para ello se debe multiplicar el error global de la red por la 
        # derivada de la función de activación de la última capa: output_error_term = (y - y_pred) * f'(x) = (y - y_pred) * 1
        output_error_term = error * 1.0
        
        # Cálculo de la contribución al error de los pesos entre la capa oculta y la capa de output. Para ello se deben multiplicar 
        # los pesos entre estas dos capas por el output_error_term: hidden_error = W_h_o * (y - y_pred) * f'(x) = 
        # = W_h_o * (y - y_pred) * 1 = W_h_o * output_error_term 
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        
        # Cálculo de la contribución al error de la capa de oculta, para ello se debe multiplicar la contribución al error de los
        # pesos entre las capas oculta y final por la derivada de la función de activación de las neuronas de la capa oculta:
        # hidden_error_term = W_h_o * output_error_term * sigmoid_prime(x) = W_h_o * (y - y_pred) * f'(x) * sigmoid_prime(x) =
        # = W_h_o * (y - y_pred) * 1 * (sigmoid(x) * (1 - sigmoid(x))) = hidden_error * (sigmoid(x) * (1 - sigmoid(x))) =
        hidden_error_term = hidden_error.T * (hidden_outputs * (1 - hidden_outputs))
        
        # Cálculo del cambio de los pesos entre los inputs y la capa oculta: X * W * (y - y_pred) * f'(x) * sigmoid_prime(x) =
        # = X * W * (y - y_pred) * 1 * (sigmoid(x) * (1 - sigmoid(x))) = X * hidden_error_term 
        delta_weights_i_h += X.reshape(-1, 1) * hidden_error_term
        
        # Cálculo del cambio de los pesos entre la capa oculta y la capa de salida: sigmoid(x) * (y - y_pred) * f'(x)  =
        # = sigmoid(x) * (y - y_pred) * 1  = hidden_outputs * output_error_term
        delta_weights_h_o += hidden_outputs.reshape(-1, 1) * output_error_term
        
        return delta_weights_i_h, delta_weights_h_o


    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Actualización de los pesos en cada paso del "gradient descent"
         
            Parámetros
            ---------
            delta_weights_i_h: cambio de los pesos entre los inputs y la capa oculta
            delta_weights_h_o: cambio de los pesos entre la capa oculta y la de salida
            n_records: número de datos
        '''
        
        # Se multiplica el cambio de los pesos entre los inputs y la capa oculta por el learning rate y luego se divide
        # el resultado entre la cantidad de datos para obtener lo que deben actualizarse los pesos a cada paso del 
        # "gradient descent": (learning_rate * hidden_outputs * output_error_term) / n_records = 
        # = (learning_rate * delta_weights_h_o) / n_records
        self.weights_hidden_to_output += (self.lr * delta_weights_h_o) / n_records 
        
        # Se multiplica el cambio de los pesos entre la capa oculta y la de salida por el learning rate y luego se divide
        # el resultado entre la cantidad de datos para obtener lo que deben actualizarse los pesos a cada paso del 
        # "gradient descent": (learning_rate * X * hidden_error_term) / n_records = 
        # = (learning_rate * delta_weights_i_h) / n_records
        self.weights_input_to_hidden += (self.lr * delta_weights_i_h) / n_records 


    def run(self, features):
        ''' Ejecución del feedforward con todos los features 
        
            Parámetros
            ---------
            features: array de una dimensión con los valores de los features
        '''
        
        # Cálculo de los valores de entrada de la capa oculta. Para ello se deben multiplicar todas las características por
        # los pesos de las conexiones que hay entre los inputs (las características) y las neuronas de la capa oculta: X * W_i_h.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) 

        # Cálculo de las salidas de las neuronas ocultas. Se utilizan los valores de entrada de cada neurona como inputs para
        # su función de activación: sigmoid(hidden_inputs) 
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        # Cálculo de los valores de entrada de la capa de salida de la red neuronal. Para ello se deben multiplicar las salidas
        # de las neuronas ocultas por los pesos que hay entre la capa de oculta y la capa de salida: hidden_outputs * W_h_o
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  
        
        # Cálculo de las salidas de las neuronas de salida de la red neuronal. Como la función de activación de las neuronas
        # de esta capa es f(x) = x, simplemente se debe igualar el output de estas neuronas a su input.
        final_outputs = final_inputs 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
# Iteraciones para completar una época:
# iterations = 1000

# Incremento de distancia en la que cambia la pendiente del gradiente de descenso:
# learning_rate = 0.1

# Número de neuronas (nodes) de la capa oculta de la red neuronal:
# hidden_nodes = 2

# Número de salidas de la red neuronal (nodes):
# output_nodes = 1

# 2o intento:
# iterations = 12000
# learning_rate = 0.5
# hidden_nodes = 23
# output_nodes = 1

# 3er intento:
# iterations = 2000
# learning_rate = 0.1
# hidden_nodes = 8
# output_nodes = 1


# 4o intento:
# iterations = 15000
# learning_rate = 0.5
# hidden_nodes = 8
# output_nodes = 1

# 5o intento: 
# iterations = 500
# learning_rate = 0.1
# hidden_nodes = 2
# output_nodes = 1

# 6o intento:
# iterations = 800
# learning_rate = 0.1
# hidden_nodes = 3
# output_nodes = 1

# 7o intento:
# iterations = 1200
# learning_rate = 0.1
# hidden_nodes = 2
# output_nodes = 1

# 8o intento:
# iterations = 12000
# learning_rate = 0.1
# hidden_nodes = 2
# output_nodes = 1

# 9o intento:
iterations = 1000
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1