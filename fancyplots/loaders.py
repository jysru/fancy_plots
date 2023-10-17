import scipy.io
import mat73

def loadmat(filepath: str) -> dict:
    try:
        loaded_dict = scipy.io.loadmat(filepath)
    except NotImplementedError:
        loaded_dict = mat73.loadmat(filepath)
    return loaded_dict
