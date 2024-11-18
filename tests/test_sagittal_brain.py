import pytest
import numpy as np

import sagittal_average.sagittal_brain


@pytest.mark.parametrize("params", [
    pytest.param(
        {
            'file_input': 'test_brain_sample.csv',
            'file_output': 'test_brain_average.csv',
            'expected_output': [i for i in range(10)]
        },
        id="positive_case_1"
    )])
def test_run_average(params: dict):
    file_input = params['file_input']
    file_output = params['file_output']
    expected_output = params['expected_output']

    output = np.loadtxt(file_output, delimiter=',', dtype=int)
    sagittal_average.sagittal_brain.run_averages(file_input, file_output)
    assert np.allclose(output, expected_output)
