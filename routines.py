import numpy as np

def get_peaks(image):
    peak_values = []
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            surrounding = (
                (x-1, y-1), (x+0, y-1), (x+1, y-1),
                (x-1, y+0),             (x+1, y+0),
                (x-1, y+1), (x+0, y+1), (x+1, y+1))
            peak = True
            for coord_pair in surrounding:
                coord_pair = (coord_pair[0] % 128, coord_pair[1] % 128)
                if image[x,y] <= image[coord_pair[0], coord_pair[1]]:
                    peak = False
                    break
            if peak:
                peak_values.append(image[x,y])
    return peak_values

try:
    from nbodykit.lab import FFTPower, ArrayMesh
    def calc_ps(field, box_size=(1000, 1000), kmin=1e-5, kmax=0.3, dk=None):
        field_mesh = ArrayMesh(field, BoxSize=box_size)
        r_2d = FFTPower(field_mesh, mode='1d', kmin=kmin, kmax=kmax, dk=dk)
        return {'power': np.real(r_2d.power['power']), 'k': np.real(r_2d.power['k'])}
except ImportError:
    try:
        import powerbox as pb
        def calc_ps(field, box_size=(1000, 1000), kmin=1e-5, kmax=0.3, dk=1e-2):
            spec, k = pb.get_power(field, box_size, bins=np.arange(kmin,kmax,dk))
            return {'power': spec, 'k': k}
    except ImportError:
        raise ImportError('Must have either NBodyKit or PowerBox installed!')
    