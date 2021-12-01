import torch
import torch.nn as nn

class Conv_Transformer_adapter(nn.Module):
    """
    This class is intended to create an interface to the Conv_Transformer that behaves like the LSTM_model.
    """
    def __init__(self, conv_transformer, configs):
        super().__init__()
        self.configs = configs
        self.conv_transformer = conv_transformer

    def forward(self, frames, baseline, prediction_count, non_pred_feat = None):
        # Conv_Transformer naturally only predicts the delta
        deltas = self.conv_transformer(frames, prediction_count, non_pred_feat)
        predictions = self.__compute_frames_from_deltas(frames, deltas, prediction_count)
        if self.configs["baseline"] == "last_frame":
            baselines = predictions
        else:
            baselines = self.__compute_means_from_frames(frames, predictions, baseline, prediction_count)

        return predictions, [deltas[..., i] for i in range(deltas.size()[-1])], baselines

    @staticmethod
    def __compute_frames_from_deltas(prev_frames, deltas, prediction_count):
        frames = []
        last_frame = prev_frames[:, :4, :, :, -1]

        for f in range(prediction_count):
            new_frame = last_frame + deltas[..., f]
            frames.append(new_frame)
            last_frame = new_frame

        return frames

    @staticmethod
    def __compute_means_from_frames(prev_frames, frames, baseline, prediction_count):
        means = [baseline]
        seq_len = prev_frames.size()[-1]

        for f in range(prediction_count):
            new_mean = 1 / (seq_len + 1) * (frames[f] + (means[-1] * seq_len))
            means.append(new_mean)
            seq_len += 1

        return means



