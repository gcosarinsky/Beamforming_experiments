import numpy as np

param_names = {'c': 'c1',
               'Bdx': 'pitch',
               'Bnx': 'n_elementos',
               'Ix0': 'x0_roi',
               'Iz0': 'z0_roi',
               'Idx': 'x_step',
               'Idz': 'z_step',
               'Inx': 'nx',
               'Inz': 'nz',
               'Cn': 'n_angles',
               'Rfs': 'fs',
               'Rnt': 'n_samples'}


def convert_to_dict(arr):
    # Extraer el primer elemento del array estructurado
    first_element = arr[0][0]

    # Crear un diccionario vacío
    result_dict = {}

    # Recorrer cada campo del array estructurado
    for field in first_element.dtype.names:
        value = first_element[field]

        # Si el valor es un array y tiene un solo elemento, extraerlo como escalar
        if isinstance(value, np.ndarray) and value.size == 1:
            result_dict[field] = value.item()
        else:
            result_dict[field] = value

    return result_dict


def dict2params_array(d):
    names = ['nx', 'n_elementos', 'n_angles', 'n_samples', 'x0_roi', 'z0_roi', 'x_step', 'z_step',
             'c1', 'pitch', 'bfd', 'fs']
    p_arr = [d[k] for k in names]
    return np.array(p_arr, dtype=np.float32)