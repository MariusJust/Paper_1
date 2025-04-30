from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
import tensorflow as tf
from .ModelFunctions import Create_dummies, create_fixed_effects, Vectorize, Count_params, Matrixize, model_with_dropout, model_without_dropout, create_output_layer, prediction_model_with_dropout, prediction_model_without_dropout, individual_loss

class RegionalBuilder:
    def __init__(
        self,
        master,
        region,
        N_region,
        T,
        noObs,
        dropout,
        time_periods_na
    ):
        # Keep reference to master for shared config/data
        self.master = master
        self.region = region
        self.N_region = N_region
        self.T = T
        self.noObs = noObs
        self.dropout = dropout
        self.time_periods_na = time_periods_na

        # 1) Define Input layers
        self.input_temp = Input(shape=(T, N_region), name=f"temp_in_{region}")
        self.input_precip = Input(shape=(T, N_region), name=f"precip_in_{region}")

        # 2) Attach training tensors
        self.x_temp = tf.reshape(
            tf.convert_to_tensor(master.x_train_transf['temp'][region]),
            (1, T, N_region)
        )
        self.x_precip = tf.reshape(
            tf.convert_to_tensor(master.x_train_transf['precip'][region]),
            (1, T, N_region)
        )
        
    
        self.mask = tf.reshape(
            tf.convert_to_tensor(master.mask[region]),
            (1, T, N_region)
        )
        self.y_true = tf.reshape(
            tf.convert_to_tensor(master.y_train_transf[region]),
            (1, T, N_region)
        )

        # 3) Loss and fixed effects
        self.loss_fn = individual_loss(mask=self.mask)
        self.delta1, self.delta2 = Create_dummies(master, self.input_temp, self.N_region, self.T, self.time_periods_na)
        self.country_fe, self.time_fe, self.country_FE_layer, self.time_FE_layer = create_fixed_effects(
            master, self.delta1, self.delta2
        )

        # 4) Vectorize & combine
        temp_vec = Vectorize(N_region, 'temp')(self.input_temp)
        precip_vec = Vectorize(N_region, 'precip')(self.input_precip)

        input_first = concatenate([temp_vec, precip_vec], axis=2)
        
        # 5) Core model body
        if dropout:
            self.last_layer = model_with_dropout(master, input_first)
        else:
            self.last_layer = model_without_dropout(master, input_first)

        # 6) Output
        out, self.output_layer = create_output_layer(master, self.last_layer)
      
        out_fe = Add()([self.time_fe, self.country_fe, out])
        self.output_matrix = Matrixize(
            N=N_region,
            T=T,
            noObs=self.noObs,
            mask=self.mask
        )(out_fe)
        
     
        


# Usage inside your main class
class Regions:
    def Setup_regional_model(self):
        self.region_builders = []

        for region in self.regions:
            builder = RegionalBuilder(
                master=self,
                region=region,
                N_region=self.N[region],
                T=self.T,
                noObs=self.noObs[region],
                dropout=self.dropout,
                time_periods_na=self.time_periods_na[region]
            )
            self.region_builders.append(builder)
            

        # extract lists 
        inputs_temp =[b.input_temp for b in self.region_builders]
        inputs_precip = [b.input_precip for b in self.region_builders]
        output_matrix = [b.output_matrix for b in self.region_builders]
        
        self.country_fe = [b.country_fe for b in self.region_builders]
        self.time_fe = [b.time_fe for b in self.region_builders]
        
        self.input_data_temp = [b.x_temp for b in self.region_builders]
        self.input_data_precip = [b.x_precip for b in self.region_builders]
        self.targets = [b.y_true for b in self.region_builders]
        self.masks = [b.mask for b in self.region_builders]
        self.loss_list = [b.loss_fn for b in self.region_builders]
        self.country_FE_layer=[b.country_FE_layer for b in self.region_builders]
        self.time_FE_layer=[b.time_FE_layer for b in self.region_builders]
        
        # 2) Build the model 
        self.model = Model(
            inputs  = [inputs_temp, inputs_precip],
            outputs = output_matrix
        )


        #count the total amount of parameters
        self.m = Count_params(self)
        
        self.model_pred = {}
        
        for b in self.region_builders:
            #    batch_input_shape must match how you'll feed it at predict‐time:
            input = Input(shape=(1, None, 2), name=f"pred_in_{b.region}")

            # 2) Reuse the core hidden layers from training
            x = self.hidden_1(input)
            if self.Depth > 1:
                x = self.hidden_2(x)
                if self.Depth > 2:
                    x = self.hidden_3(x)

            # 3) Apply the region's output layer
            y = b.output_layer(x)

            # 4) Wrap into its own Model
            self.model_pred[b.region] = Model(inputs=input, outputs=y,
                                              name=f"pred_model_{b.region}")

            
            
            # if self.dropout:
            #     self.model_pred[b.region] = prediction_model_with_dropout(self, input)
            # else:
            #     self.model_pred[b.region] = prediction_model_without_dropout(self, input)
                