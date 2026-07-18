# -*- coding: utf-8 -*-
"""
==========================================================================================
Core Model Definitions: SwinTrans-ConvLSTM & Physics-Informed Curriculum Loss
Manuscript: "Deep Learning-Based Prediction of Marine Heatwaves in the East China Sea"
==========================================================================================
"""
import torch
import torch.nn as nn
from einops import rearrange

# ==============================================================================
# --- 1. ConvLSTM Components for Temporal Evolution ---
# ==============================================================================
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
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
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
        if not self.batch_first: 
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None: 
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
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
            
        if not self.return_all_layers: 
            layer_output_list, last_state_list = layer_output_list[-1:], last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.cell_list[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            return [param] * num_layers
        return param

# ==============================================================================
# --- 2. Swin Transformer Components for Spatial Representation ---
# ==============================================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads, head_dim = dim, window_size, num_heads, dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # Relative position bias table
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
        
        # Cyclic shift
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        
        # Partition windows
        x_windows = rearrange(shifted_x, 'b (h h_w) (w w_w) c -> (b h w) (h_w w_w) c', h_w=self.window_size, w_w=self.window_size)
        
        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=None)
        
        # Merge windows
        shifted_x = rearrange(attn_windows, '(b h w) (h_w w_w) c -> b (h h_w) (w w_w) c', h=H // self.window_size,
                              w=W // self.window_size, h_w=self.window_size, w_w=self.window_size)
        
        # Reverse cyclic shift
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        
        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x

# ==============================================================================
# --- 3. Hybrid Model: SwinTrans-ConvLSTM ---
# ==============================================================================
class SwinConvLSTMModel(nn.Module):
    def __init__(self, input_dim, spatial_dims, pred_steps, convlstm_hidden_dim=64, convlstm_kernel_size=3,
                 convlstm_num_layers=1, d_model=96, n_heads=[3, 6], num_encoder_layers=2, window_size=10, dropout=0.2):
        super().__init__()
        self.H, self.W = spatial_dims
        self.pred_steps = pred_steps
        
        # 1. Local Temporal Evolution Extraction
        self.temporal_encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=convlstm_hidden_dim,
            kernel_size=(convlstm_kernel_size, convlstm_kernel_size),
            num_layers=convlstm_num_layers,
            batch_first=True,
            bias=True
        )
        
        # Projection and Positional Embedding
        self.input_proj = nn.Conv2d(convlstm_hidden_dim, d_model, kernel_size=1)
        self.pos_embed = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        # 2. Global Spatial Representation Extraction
        self.encoder_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=d_model,
                num_heads=n_heads[i % len(n_heads)],
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            ) for i in range(num_encoder_layers)
        ])
        
        # 3. Final Prediction Decoder
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

    def forward(self, x):
        # Input shape: [Batch, Height, Width, Time, Features]
        B, H, W, T, F = x.shape
        
        # Reshape for ConvLSTM: [Batch, Time, Features, Height, Width]
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        
        # Temporal encoding
        _, last_state_list = self.temporal_encoder(x)
        x_temporal_encoded = last_state_list[-1][0]
        
        # Projection & Embedding
        x_proj = self.input_proj(x_temporal_encoded)
        pos_feat = self.pos_embed(x_proj)
        x = x_proj.flatten(2).transpose(1, 2) + pos_feat.flatten(2).transpose(1, 2)
        
        # Swin Transformer Blocks
        for block in self.encoder_blocks:
            x = block(x, H, W)
            
        # Decode to forecast steps
        x = x.transpose(1, 2).view(B, -1, H, W)
        prediction_raw = self.decoder(x)
        
        # Output shape: [Batch, Height, Width, Pred_Steps]
        return prediction_raw.permute(0, 2, 3, 1).contiguous()

# ==============================================================================
# --- 4. Custom Physics-Informed Curriculum Loss ---
# ==============================================================================
class PhysicsCurriculumLoss(nn.Module):
    def __init__(self, start_ohem_epoch=20, ohem_ratio=0.2, grad_weight=1.0):
        super().__init__()
        self.start_ohem_epoch = start_ohem_epoch
        self.ohem_ratio = ohem_ratio
        self.grad_weight = grad_weight
        self.criterion = nn.HuberLoss(reduction='none')

    def _compute_spatial_gradient(self, image):
        dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        dx = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        return dy, dx

    def forward(self, outputs, targets, seasonal_weights, current_epoch):
        # 1. Base Element-wise Loss with Seasonal Weighting
        loss_per_element = self.criterion(outputs, targets)
        weighted_loss = loss_per_element * seasonal_weights

        # 2. Curriculum Learning Strategy: Global Mean -> OHEM
        if current_epoch < self.start_ohem_epoch:
            base_loss = weighted_loss.mean()
        else:
            # Online Hard Example Mining (OHEM) for extreme events
            loss_flat = weighted_loss.reshape(-1)
            k = int(loss_flat.numel() * self.ohem_ratio)
            k = max(k, 1)
            topk_loss, _ = torch.topk(loss_flat, k)
            base_loss = topk_loss.mean()

        # 3. Physics-Informed Spatial Gradient Constraint
        grad_pred_y, grad_pred_x = self._compute_spatial_gradient(outputs)
        grad_true_y, grad_true_x = self._compute_spatial_gradient(targets)
        
        loss_grad_y = nn.L1Loss()(grad_pred_y, grad_true_y)
        loss_grad_x = nn.L1Loss()(grad_pred_x, grad_true_x)
        gradient_loss = self.grad_weight * (loss_grad_y + loss_grad_x)

        return base_loss + gradient_loss