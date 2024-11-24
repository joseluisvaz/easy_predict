import torch
import torch.nn as nn
from torch_scatter import scatter_max


class ContextGating(nn.Module):
    def __init__(self, hidden_size: int, is_identity: bool = False):
        super(ContextGating, self).__init__()
        self.linear_features = nn.Linear(hidden_size, hidden_size)
        self.linear_context = nn.Identity() if is_identity else nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden: torch.Tensor,
        context_embedding: torch.Tensor,
        availabilities: torch.Tensor,
    ):
        """
        hidden of size (batch, agents, features)
        context_embeddig (batch, features)
        availabilities (batch, agents)

        Returns the embeddings of size (batch, agents, features) and the context of size (batch, features)
        """
        assert hidden.dim() == 3 and context_embedding.dim() == 2

        embeddings = self.linear_features(hidden)
        context = self.linear_context(context_embedding)
        embeddings = embeddings * context.unsqueeze(1)


        # TODO make the max pooling a utility function
        batch_size, _, output_size = embeddings.shape
        arange = torch.arange(0, batch_size, device=hidden.device)
        expanded_range = arange.unsqueeze(1).expand(-1, hidden.size(1))  # (batch, agents)

        scatter_index = expanded_range[availabilities]
        embeddings_flattened = embeddings[availabilities].view(-1, output_size)
        context, _ = scatter_max(embeddings_flattened, scatter_index, dim=0)
        
        return embeddings, context


class MultiContextGating(nn.Module):
    def __init__(self, hidden_size: int, n_contexts: int):
        super(MultiContextGating, self).__init__()
        self.n_contexts = n_contexts

        context_gatings = []
        for idx in range(n_contexts):
            context_gatings.append(ContextGating(hidden_size, is_identity=(idx == 0)))

        self.context_gatings = nn.ModuleList(context_gatings)

    def _get_initial_context(self, batch_size, hidden_size, device):
        """Returns a trainable initial context of shape: (batch_size, hidden_size)"""
        return torch.ones(batch_size, hidden_size, device=device, requires_grad=True)

    def _compute_running_mean(self, previous_mean, new_value, i):
        return (previous_mean * i + new_value) / i

    def forward(self, hidden, availabilities):
        """
        hidden has shape (batch_size, n_agents, hidden_size)
        availabilities has shape (batch_size, n_agents,)


        Returns the mean of the hidden states after applying the context gating mechanism

        The return tensor has shape (batch_size, n_agents, hidden_size)
        """
        assert hidden.dim() == 3
        context = self._get_initial_context(hidden.size(0), hidden.size(2), hidden.device)

        previous_hidden_mean = hidden
        previous_context_mean = context
        for idx in range(self.n_contexts):
            hidden, context = self.context_gatings[idx](
                previous_hidden_mean, previous_context_mean, availabilities
            )

            previous_hidden_mean = self._compute_running_mean(previous_hidden_mean, hidden, idx + 1)
            previous_context_mean = self._compute_running_mean(
                previous_context_mean, context, idx + 1
            )

        return previous_hidden_mean
