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


class KernelParameters:
    # Definir listas de nombres para cada tipo de parámetro
    float_names = ['fs', 'c1', 'pitch', 'f1', 'f2', 'bfd', 'x_step', 'z_step',
                   'x0_roi', 'z0_roi', 't_start', 'x_0']
    int_names = ['taps', 'n_elementos', 'n_angles', 'nx', 'nz', 'n_samples']

    def __init__(self, params_dict):
        # Verificar que todos los parámetros necesarios estén presentes
        missing_params = []
        for name in self.float_names + self.int_names:
            if name not in params_dict:
                missing_params.append(name)

        if missing_params:
            raise ValueError(f"Faltan los siguientes parámetros: {', '.join(missing_params)}")

        # Asignar valores float
        for name in self.float_names:
            setattr(self, name, float(params_dict[name]))

        # Asignar valores int
        for name in self.int_names:
            setattr(self, name, int(params_dict[name]))

    def get_float_array(self):
        """Devuelve array ordenado de parámetros float"""
        return np.array([getattr(self, name) for name in self.float_names],
                        dtype=np.float32)

    def get_int_array(self):
        """Devuelve array ordenado de parámetros int"""
        return np.array([getattr(self, name) for name in self.int_names],
                        dtype=np.int32)