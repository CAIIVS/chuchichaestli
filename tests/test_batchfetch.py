"""Tests for the fetch module.

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

import pytest
import torch

from chuchichaestli.injectables.batchfetch import (
    ExtractD,
    ExtractS,
    Identity,
    SubsetD,
    SubsetS,
)


def test_identity():
    identity = Identity()
    input_data = [1, 2, 3]
    assert identity(input_data) == input_data

    input_data = {"a": 1, "b": 2}
    assert identity(input_data) == input_data

    input_tensor = torch.tensor([1, 2, 3])
    assert torch.equal(identity(input_tensor), input_tensor)


def test_extract_d_single():
    extract_d = ExtractD("key1")
    input_dict = {"key1": torch.tensor([1, 2, 3]), "key2": torch.tensor([4, 5, 6])}
    assert torch.equal(extract_d(input_dict), torch.tensor([1, 2, 3]))

    with pytest.raises(KeyError):
        extract_d({"key3": torch.tensor([7, 8, 9])})

def test_extract_d_multiple():
    extract_d = ExtractD(["key1", "key2"])
    input_dict = {"key1": torch.tensor([1, 2, 3]), "key2": torch.tensor([4, 5, 6])}
    assert torch.equal(extract_d(input_dict)[0], torch.tensor([1, 2, 3]))
    assert torch.equal(extract_d(input_dict)[1], torch.tensor([4, 5, 6]))

    with pytest.raises(KeyError):
        extract_d({"key3": torch.tensor([7, 8, 9])})

def test_subset_d():
    subset_d_single = SubsetD("key1")
    subset_d_multiple = SubsetD(["key1", "key2"])
    
    input_dict = {
        "key1": torch.tensor([1, 2, 3]),
        "key2": torch.tensor([4, 5, 6]),
        "key3": torch.tensor([7, 8, 9])
    }

    result_single = subset_d_single(input_dict)
    assert len(result_single) == 1
    assert "key1" in result_single
    assert torch.equal(result_single["key1"], torch.tensor([1, 2, 3]))

    result_multiple = subset_d_multiple(input_dict)
    assert len(result_multiple) == 2
    assert "key1" in result_multiple and "key2" in result_multiple
    assert torch.equal(result_multiple["key1"], torch.tensor([1, 2, 3]))
    assert torch.equal(result_multiple["key2"], torch.tensor([4, 5, 6]))

    with pytest.raises(KeyError):
        SubsetD(["key1", "key4"])(input_dict)
    

def test_extract_s():
    extract_s = ExtractS(1)
    input_sequence = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
    assert torch.equal(extract_s(input_sequence), torch.tensor([4, 5, 6]))

    with pytest.raises(IndexError):
        extract_s([torch.tensor([1, 2, 3])])


def test_subset_s():
    subset_s_single = SubsetS(1)
    subset_s_multiple = SubsetS([0, 2])
    
    input_sequence = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6]),
        torch.tensor([7, 8, 9])
    ]
    
    assert len(subset_s_single(input_sequence)) == 1
    assert torch.equal(subset_s_single(input_sequence)[0], torch.tensor([4, 5, 6]))
    
    assert len(subset_s_multiple(input_sequence)) == 2
    assert torch.equal(subset_s_multiple(input_sequence)[0], torch.tensor([1, 2, 3]))
    assert torch.equal(subset_s_multiple(input_sequence)[1], torch.tensor([7, 8, 9]))

    with pytest.raises(IndexError):
        SubsetS([0, 3])(input_sequence)

