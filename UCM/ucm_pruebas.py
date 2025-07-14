import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py


def leer_param_y_data(mat_file, param_group='param', data_key='data'):
    """
    Lee todos los arrays referenciados en un archivo .mat v7.3 en el campo 'data' y el resto de parámetros en 'param'.

    Args:
        mat_file: ruta al archivo .mat
        param_group: nombre del grupo principal (por defecto 'param')
        data_key: nombre del campo con referencias (por defecto 'data')
    Returns:
        Un diccionario con:
            'data': lista de arrays (numpy.ndarray) leídos desde las referencias
            otros campos: valores de los parámetros escalares o arrays
    """
    result = {}
    with h5py.File(mat_file, 'r') as file:
        param = file[param_group]

        # Lee los datos referenciados
        refs = param[data_key][:]
        it = refs.flat if hasattr(refs, 'flat') else refs.ravel()
        acq = []
        for ref in it:
            if ref:
                arr = file[ref][:]
                acq.append(arr)
            else:
                acq.append(None)
        result[data_key] = acq

        # Lee el resto de los parámetros del grupo
        for key in param.keys():
            if key == data_key:
                continue
            obj = param[key]
            if isinstance(obj, h5py.Dataset):
                try:
                    val = obj[()]
                    # Si es un array de bytes, decodifica (por si algunos strings se guardan como bytes)
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    result[key] = val
                except Exception as e:
                    result[key] = None
            elif isinstance(obj, h5py.Group):
                # Si hay grupos anidados, puedes guardar sus keys o hacer una lectura recursiva si lo necesitas
                result[key] = {k: obj[k][()] for k in obj.keys()}
            else:
                result[key] = None
    return result


# Ejemplo de uso:
if __name__ == "__main__":
    mat_file = \
        r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\UCM\acquisiton_files\25-07-11_13-03-28_prueba_18MHZ_PWI\PWI\prueba_18MHZ_PWI_PWI_25-07-11_13-03-28.mat'
    param_dic = leer_param_y_data(mat_file)
    print(param_dic.keys())
    # Ejemplo: imprime el valor de 'fs'
    print("fs:", param_dic.get('fs'))
    # Ejemplo: primer array de 'data'
    print("Primer array shape:", param_dic['data'][0].shape if param_dic['data'][0] is not None else None)

    n_acq = int(param_dic['n_acquisitions'][0][0])

    matrix_shape = \
        (int(param_dic['n_focal_laws'][0][0]), int(param_dic['n_ascan'][0][0]), int(param_dic['n_samples'][0][0]))
    acq = np.empty((n_acq, ) + matrix_shape, dtype=np.int16)
    for i in range(n_acq):
        acq[i, ...] = param_dic['data'][i].reshape(matrix_shape)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))  # Crear 1 fila y 4 columnas de subplots
    for i in range(4):  # Iterar sobre los primeros 4 índices
        ax = axes[i]
        ax.imshow(np.abs(acq[0, i, :, :].T))  # Pintar la matriz correspondiente
        ax.set_aspect(0.1)  # Ajustar el aspecto
        ax.set_title(f"Índice {i}")  # Título para cada subplot