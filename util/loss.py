import tensorflow as tf

def individual_loss(mask):
    """
    Loss function (in two layers so that it can be interpreted by tensorflow).

    ARGUMENTS
        * mask: mask used to identify missing observations.

    Returns
        * loss: loss function evaluated in y_true and y_pred.
    """

    def loss(y_true, y_pred):
        """
        ARGUMENTS
            * y_true: observed targets.
            * y_pred: predicted targets.

        RETURNS
            * loss function evaluated in y_true and y_pred.
        """

        y_true_transf = tf.reshape(y_true[~mask], (1, -1, 1))
        y_pred_transf = tf.reshape(y_pred[~mask], (1, -1, 1))
        loss=tf.reduce_mean(tf.math.squared_difference(y_true_transf, y_pred_transf), axis=1)
        return loss

    return loss
