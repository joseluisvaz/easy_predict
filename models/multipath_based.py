import torch.nn as nn
from models.multi_context_gating import MultiContextGating
from models.rnn_cells import MultiAgentLSTMCell


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequence, mask):
        n_batch, n_agents, n_timesteps, _ = sequence.shape
        hidden, context = self.lstm_cell.get_initial_hidden_state(
            (n_batch, n_agents, self.hidden_size), sequence.device, requires_grad=True
        )

        for t in range(n_timesteps):
            input_t = sequence[:, :, t, :]
            mask_t = mask[:, :, t]
            hidden, context = self.lstm_cell(input_t, (hidden, context), mask_t)

        return hidden, context


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_timesteps):
        super(Decoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.n_timesteps = n_timesteps

    def forward(self, current_positions, current_availabilities, hidden, context):
        output = current_positions

        outputs = []
        for t in range(self.n_timesteps):
            hidden, context = self.lstm_cell(output, (hidden, context), current_availabilities)
            delta_output = self.linear(hidden)
            output = output + delta_output
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)
        return outputs


class MultiPathBased(nn.Module):

    XY_OUTPUT_SIZE = 2
    NUM_MCG_LAYERS = 4

    def __init__(self, input_features, hidden_size, n_timesteps):
        super(MultiPathBased, self).__init__()
        self.encoder = Encoder(input_features, hidden_size)
        self.mcg = MultiContextGating(hidden_size, n_contexts=self.NUM_MCG_LAYERS)
        self.decoder = Decoder(self.XY_OUTPUT_SIZE, hidden_size, self.XY_OUTPUT_SIZE, n_timesteps)

    def forward(self, history_features, history_availabilities):
        hidden, context = self.encoder(history_features, history_availabilities)
        current_positions = history_features[:, :, -1, :2]  # For now only takes positions
        current_availabilities = history_availabilities[:, :, -1]

        hidden = self.mcg(hidden, current_availabilities)

        output = self.decoder(current_positions, current_availabilities, hidden, context)
        return output
