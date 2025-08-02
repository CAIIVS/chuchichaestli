"""Loss & evaluation metric implementations.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

from chuchichaestli.metrics.mse import MSE
from chuchichaestli.metrics.psnr import PSNR
from chuchichaestli.metrics.ssim import SSIM, SSIMLoss
from chuchichaestli.metrics.fid import FID
from chuchichaestli.metrics.lpips import LPIPSLoss


__all__ = ["MSE", "PSNR", "SSIM", "SSIMLoss", "FID", "LPIPSLoss"]
