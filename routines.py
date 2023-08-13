from nbodykit.lab import FFTPower, ArrayMesh

## Davide's routine to calculate a power spectrum using NBK

kmin = 1e-5 # in h/Mpc
kmax = 0.3 # in h/Mpc # apparently higher values fail

def calc_ps(field, BoxSize=[1000, 1000], kmin=kmin, kmax=kmax, dk=1e-2):
    field_mesh = ArrayMesh(field, BoxSize=BoxSize)
    r_2d = FFTPower(field_mesh, mode='1d', kmin=kmin, kmax=kmax)#, dk=1e-4)
    return r_2d.power

## Liam's adapted routine to find peaks

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