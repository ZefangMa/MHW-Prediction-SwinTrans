"""SwinTrans-ConvLSTM model components for direct multi-lead SST forecasting."""

import torch
from torch import nn
from einops import rearrange


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True,
                 return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim

        self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(self.kernel_size) == len(self.hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first: input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, seq_len, _, h, w = input_tensor.size()
        if hidden_state is None: hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        layer_output_list, last_state_list = [], []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers: layer_output_list, last_state_list = layer_output_list[-1:], last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.cell_list[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        candidates = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        for kernel in candidates:
            sizes = (kernel, kernel) if isinstance(kernel, int) else kernel
            if (not isinstance(sizes, tuple) or len(sizes) != 2
                    or any(not isinstance(k, int) or k < 1 or k % 2 == 0 for k in sizes)):
                raise ValueError("ConvLSTM kernels must be positive odd integers.")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            return [param] * num_layers
        return param


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads, head_dim = dim, window_size, num_heads, dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, ffn_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * ffn_ratio)), nn.GELU(), nn.Linear(int(dim * ffn_ratio), dim))
        self.shift_size, self.window_size = shift_size, window_size

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                               dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = rearrange(shifted_x, 'b (h h_w) (w w_w) c -> (b h w) (h_w w_w) c', h_w=self.window_size,
                              w_w=self.window_size)
        attn_windows = self.attn(x_windows, mask=None)
        shifted_x = rearrange(attn_windows, '(b h w) (h_w w_w) c -> b (h h_w) (w w_w) c', h=H // self.window_size,
                              w=W // self.window_size, h_w=self.window_size, w_w=self.window_size)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                       dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x


class SwinConvLSTMModel(nn.Module):
    """Jointly predict all lead days from one [batch, lat, lon, time, feature] tensor.

    Returns [batch, lat, lon, lead]. This module learns in the target units
    provided during training; the demo uses standardized SST anomalies.
    """

    def __init__(self, input_dim, spatial_dims, pred_steps=7, convlstm_hidden_dim=64,
                 convlstm_kernel_size=3, convlstm_num_layers=1, d_model=96,
                 n_heads=(3, 6), num_encoder_layers=2, window_size=10, dropout=0.2):
        super().__init__()
        self.H, self.W = spatial_dims
        self.input_dim = input_dim
        self.pred_steps = pred_steps
        if window_size < 1 or min(self.H, self.W) < 1 or self.H % window_size or self.W % window_size:
            raise ValueError("Grid dimensions must be positive multiples of the attention window.")
        if not n_heads or any(head < 1 or d_model % head for head in n_heads):
            raise ValueError("Each attention head count must be a positive divisor of d_model.")
        if min(input_dim, pred_steps, convlstm_hidden_dim, convlstm_num_layers, num_encoder_layers) < 1:
            raise ValueError("Channel counts, lead count, and layer counts must be positive.")
        ConvLSTM._check_kernel_size_consistency(convlstm_kernel_size)
        # dropout is retained for compatibility with the original constructor.
        self.temporal_encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=convlstm_hidden_dim,
            kernel_size=(convlstm_kernel_size, convlstm_kernel_size),
            num_layers=convlstm_num_layers,
            batch_first=True,
            bias=True
        )
        self.input_proj = nn.Conv2d(convlstm_hidden_dim, d_model, kernel_size=1)
        self.pos_embed = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.encoder_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=d_model,
                num_heads=n_heads[i % len(n_heads)],
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            ) for i in range(num_encoder_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model * 2, pred_steps, kernel_size=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, ConvLSTMCell):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:3] != (self.H, self.W) or x.shape[-1] != self.input_dim:
            raise ValueError("Expected inputs shaped [batch, latitude, longitude, time, feature].")
        B, H, W, T, _ = x.shape
        if T < 1:
            raise ValueError("Provide at least one historical input day.")
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        _, last_state_list = self.temporal_encoder(x)
        x_temporal_encoded = last_state_list[-1][0]
        x_proj = self.input_proj(x_temporal_encoded)
        pos_feat = self.pos_embed(x_proj)
        x = x_proj.flatten(2).transpose(1, 2) + pos_feat.flatten(2).transpose(1, 2)
        for block in self.encoder_blocks:
            x = block(x, H, W)
        x = x.transpose(1, 2).view(B, -1, H, W)
        prediction_raw = self.decoder(x)
        return prediction_raw.permute(0, 2, 3, 1).contiguous()


class CurriculumGradientLoss(nn.Module):
    """Apply Huber loss, optional hard-example mining, and spatial-gradient regularization.

    Predictions and targets are [batch, lead, latitude, longitude].
    The ocean mask is [latitude, longitude], with True for ocean.
    Nonfinite targets and land are excluded, including from gradient pairs.
    """

    def __init__(self, ocean_mask, ohem_start_epoch=20, ohem_ratio=0.20, gradient_weight=1.0):
        super().__init__()
        if not 0 < ohem_ratio <= 1 or ohem_start_epoch < 0 or gradient_weight < 0:
            raise ValueError("Invalid hard-example mining or gradient weight setting.")
        ocean_mask = torch.as_tensor(ocean_mask, dtype=torch.bool)
        if ocean_mask.ndim != 2 or not bool(ocean_mask.any()):
            raise ValueError("ocean_mask must be a two-dimensional mask with at least one ocean cell.")
        self.register_buffer("ocean_mask", ocean_mask[None, None])
        self.ohem_start_epoch = ohem_start_epoch
        self.ohem_ratio = ohem_ratio
        self.gradient_weight = gradient_weight
        self.huber = nn.HuberLoss(delta=1.0, reduction="none")

    @staticmethod
    def _masked_average(values, valid):
        if not bool(valid.any()):
            return values.sum() * 0.0
        return values[valid].mean()

    def forward(self, pred_sst, true_sst, seasonal_weights=1.0, epoch=0):
        if (pred_sst.ndim != 4 or pred_sst.shape != true_sst.shape
                or pred_sst.shape[-2:] != self.ocean_mask.shape[-2:]):
            raise ValueError("Predictions and targets must match [batch, lead, latitude, longitude].")
        valid = torch.isfinite(true_sst) & self.ocean_mask
        if not bool(valid.any()):
            raise ValueError("This batch has no valid ocean targets.")
        if not bool(torch.isfinite(pred_sst[valid]).all()):
            raise FloatingPointError("Predictions must be finite where ocean targets are valid.")
        true_safe = torch.where(valid, true_sst, torch.zeros_like(true_sst))
        pred_safe = torch.where(valid, pred_sst, torch.zeros_like(pred_sst))
        element_loss = self.huber(pred_safe, true_safe) * seasonal_weights
        valid_losses = element_loss[valid]
        if epoch < self.ohem_start_epoch:
            base_loss = valid_losses.mean()
        else:
            k = max(1, int(valid_losses.numel() * self.ohem_ratio))
            base_loss = torch.topk(valid_losses, k).values.mean()
        # Only include adjacent pairs of valid ocean targets.
        dy_valid = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        dx_valid = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        dy_pred = torch.abs(pred_safe[:, :, 1:, :] - pred_safe[:, :, :-1, :])
        dy_true = torch.abs(true_safe[:, :, 1:, :] - true_safe[:, :, :-1, :])
        dx_pred = torch.abs(pred_safe[:, :, :, 1:] - pred_safe[:, :, :, :-1])
        dx_true = torch.abs(true_safe[:, :, :, 1:] - true_safe[:, :, :, :-1])
        gradient = self._masked_average(torch.abs(dy_pred - dy_true), dy_valid)
        gradient = gradient + self._masked_average(torch.abs(dx_pred - dx_true), dx_valid)
        return base_loss + self.gradient_weight * gradient


@torch.no_grad()
def predict_sst(model, inputs, reference_sst, target_mean, target_std, ocean_mask, *, flip_average=False):
    """Return SST in Celsius as [batch, lead, latitude, longitude].

    Pass a model in evaluation mode and training-derived reference statistics
    broadcastable to the forecast grid. The mask marks ocean with True.
    Optional longitude flip averaging uses only the same historical inputs.
    """
    standardized = model(inputs).permute(0, 3, 1, 2)
    if flip_average:
        flipped = model(inputs.flip(dims=(2,))).permute(0, 3, 1, 2)
        standardized = (standardized + flipped.flip(dims=(3,))) / 2

    standardized = standardized.float()
    def as_forecast_tensor(value):
        return torch.as_tensor(value, dtype=standardized.dtype, device=standardized.device)

    reference_sst = as_forecast_tensor(reference_sst)
    target_mean = as_forecast_tensor(target_mean)
    target_std = as_forecast_tensor(target_std)
    ocean_mask = torch.as_tensor(ocean_mask, dtype=torch.bool, device=standardized.device)
    if ocean_mask.shape != standardized.shape[-2:]:
        raise ValueError("The ocean mask must match the forecast grid.")
    sst = standardized * target_std + target_mean + reference_sst
    if sst.shape != standardized.shape:
        raise ValueError("Reference statistics must broadcast to [batch, lead, latitude, longitude].")
    return sst.masked_fill(~ocean_mask, float("nan"))
