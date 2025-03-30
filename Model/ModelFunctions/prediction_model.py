from tensorflow.keras import Model

def setup_prediction_model(self, input_x_pred):
    """
    Setting up the prediction model.

    ARGUMENTS
        * input_x_pred: input tensor for the prediction model.

    RETURNS
        * model_pred: prediction model.
    """

    hidden_1_pred = self.hidden_1(input_x_pred)

    if self.Depth > 1:
        hidden_2_pred = self.hidden_2(hidden_1_pred)

        if self.Depth > 2:
            hidden_3_pred = self.hidden_3(hidden_2_pred)
            input_last_pred = hidden_3_pred

        else:
            input_last_pred = hidden_2_pred

    else:
        input_last_pred = hidden_1_pred

    output_pred = self.output_layer(input_last_pred)

    model_pred = Model(inputs=input_x_pred, outputs=output_pred)

    return model_pred