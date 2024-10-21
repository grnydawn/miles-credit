
import torch
import torch.nn.functional as F

class TensorPadding:
    def __init__(self, conf_padding):
        '''
        Initialize the TensorPadding class with the specified mode and padding sizes.

        Args:
            mode (str): The padding mode, either 'mirror' or 'earth'.
            pad_NS (list[int]): Padding sizes for the North-South (latitude) dimension [top, bottom].
            pad_WE (list[int]): Padding sizes for the West-East (longitude) dimension [left, right].
        '''
        
        self.mode = conf_padding['mode']
        self.pad_NS = conf_padding['pad_lat']
        self.pad_WE = conf_padding['pad_lon']
        

    def pad(self, x):
        '''
        Apply padding to the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The padded tensor.
        '''
        if self.mode == 'mirror':
            return self._mirror_padding(x)
        elif self.mode == 'earth':
            return self._earth_padding(x)

    def unpad(self, x):
        '''
        Remove padding from the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Padded tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The unpadded tensor.
        '''
        if self.mode == 'mirror':
            return self._mirror_unpad(x)
        elif self.mode == 'earth':
            return self._earth_unpad(x)

    def _earth_padding(self, x):
        '''
        Apply earth padding to the tensor (poles and circular padding).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        '''
        if any(p > 0 for p in self.pad_NS):
            # 180-degree shift using half the longitude size
            shift_size = int(x.shape[-1] // 2)
            xroll = torch.roll(x, shifts=shift_size, dims=-1)
            # pad poles
            xroll_flip_top = torch.flip(xroll[..., :self.pad_NS[0], :], (-2,))
            xroll_flip_bot = torch.flip(xroll[..., -self.pad_NS[1]:, :], (-2,))
            x = torch.cat([xroll_flip_top, x, xroll_flip_bot], dim=-2)

        if any(p > 0 for p in self.pad_WE):
            x = F.pad(x, (self.pad_WE[0], self.pad_WE[1], 0, 0, 0, 0), mode='circular')

        return x

    def _earth_unpad(self, x):
        '''
        Remove earth padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        '''
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            start_NS = self.pad_NS[0]
            end_NS = -self.pad_NS[1] if self.pad_NS[1] > 0 else None
            x = x[..., start_NS:end_NS, :]

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            start_WE = self.pad_WE[0]
            end_WE = -self.pad_WE[1] if self.pad_WE[1] > 0 else None
            x = x[..., :, start_WE:end_WE]

        return x

    def _mirror_padding(self, x):
        '''
        Apply mirror padding to the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        '''
        # pad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            pad_lon_left, pad_lon_right = self.pad_WE
            x = F.pad(x, pad=(pad_lon_left, pad_lon_right, 0, 0, 0, 0), mode='circular')

        # pad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            x_shape = x.shape  # (batch, var, time, lat, lon)
            x = x.reshape(x_shape[0], x_shape[1] * x_shape[2], x_shape[3], x_shape[4])
            pad_lat_top, pad_lat_bottom = self.pad_NS
            x = F.pad(x, pad=(0, 0, pad_lat_top, pad_lat_bottom, 0, 0), mode='reflect')
            new_lat = x_shape[3] + pad_lat_top + pad_lat_bottom
            x = x.reshape(x_shape[0], x_shape[1], x_shape[2], new_lat, x_shape[4])

        return x

    def _mirror_unpad(self, x):
        '''
        Remove mirror padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        '''
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            pad_lat_top, pad_lat_bottom = self.pad_NS
            start_NS = pad_lat_top
            end_NS = -pad_lat_bottom if pad_lat_bottom > 0 else None
            x_shape = x.shape
            x = x.reshape(x_shape[0], x_shape[1] * x_shape[2], x_shape[3], x_shape[4])
            x = x[..., start_NS:end_NS, :]
            new_lat = x.shape[2]
            x = x.reshape(x_shape[0], x_shape[1], x_shape[2], new_lat, x_shape[4])

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            pad_lon_left, pad_lon_right = self.pad_WE
            start_WE = pad_lon_left
            end_WE = -pad_lon_right if pad_lon_right > 0 else None
            x = x[..., :, start_WE:end_WE]

        return x