#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
#include "numpy/arrayobject.h"

//#include "PTripepClosure.h"

#define NEAREST_INT(a) ((int) ((a) + (0.5)))


static inline int modulo(int x, int N)
{
  int ret = x % N;
  if (ret < 0)
    ret += N;
  return ret;
}

static PyObject *dilate_points(PyObject *dummy, PyObject *args)
{
    // Parse arguments
    PyObject *py_points_arg=NULL,
             *py_active_arg=NULL,
             *py_occupancies_arg=NULL,
             *py_lmax_arg=NULL,
             *py_radial_densities_arg=NULL,
             *py_grid_to_cartesian_arg=NULL,
             *py_out_arg=NULL;
    double rstep, rmax;
    PyArrayObject *py_points=NULL,
                  *py_active=NULL,
                  *py_occupancies=NULL,
                  *py_lmax=NULL,
                  *py_radial_densities=NULL,
                  *py_grid_to_cartesian=NULL,
                  *py_out=NULL;

    if (!PyArg_ParseTuple(args, "OOOOOddOO",
                &py_points_arg,
                &py_active_arg,
                &py_occupancies_arg,
                &py_lmax_arg,
                &py_radial_densities_arg,
                &rstep,
                &rmax,
                &py_grid_to_cartesian_arg,
                &py_out_arg))
        return NULL;

    py_points = (PyArrayObject *)
        PyArray_FROM_OTF(py_points_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_points == NULL)
        goto fail;

    py_active = (PyArrayObject *)
        PyArray_FROM_OTF(py_active_arg, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (py_active == NULL)
        goto fail;

    py_occupancies = (PyArrayObject *)
        PyArray_FROM_OTF(py_occupancies_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_points == NULL)
        goto fail;

    py_lmax = (PyArrayObject *)
        PyArray_FROM_OTF(py_lmax_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_lmax == NULL)
        goto fail;

    py_radial_densities = (PyArrayObject *)
        PyArray_FROM_OTF(py_radial_densities_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_radial_densities == NULL)
        goto fail;

    py_grid_to_cartesian = (PyArrayObject *)
        PyArray_FROM_OTF(py_grid_to_cartesian_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_grid_to_cartesian == NULL)
        goto fail;

    py_out = (PyArrayObject *)
        PyArray_FROM_OTF(py_out_arg, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY2);
    if (py_out == NULL)
        goto fail;

    // Get pointers to arrays and shape info.
    double *points = (double *) PyArray_DATA(py_points);
    npy_bool *active = (npy_bool *) PyArray_DATA(py_active);
    double *radial_densities = (double *) PyArray_DATA(py_radial_densities);
    double *lmax = (double *) PyArray_DATA(py_lmax);
    double *occupancies = (double *) PyArray_DATA(py_occupancies);
    double *grid_to_cartesian = (double *) PyArray_DATA(py_grid_to_cartesian);
    double *out = (double *) PyArray_DATA(py_out);

    npy_intp *points_shape = PyArray_DIMS(py_points);
    npy_intp *radial_densities_shape = PyArray_DIMS(py_radial_densities);
    npy_intp *out_shape = PyArray_DIMS(py_out);

    // After the parsing and generation of pointers, the real function can
    // start.

    // Precompute some values.
    int out_slice = out_shape[2] * out_shape[1];
    int out_size = out_slice * out_shape[0];
    double rmax2 = rmax * rmax;

    int n;
    for (n = 0; n < points_shape[0]; n++) {

        if (!active[n]) {
            continue;
        }

        double q = occupancies[n];
        int radial_densities_ind = n * radial_densities_shape[1];
        size_t point_ind = 3 * n;
        double center_a = points[point_ind];
        double center_b = points[point_ind + 1];
        double center_c = points[point_ind + 2];

        int cmin = (int) ceil(center_c - lmax[2]);
        int bmin = (int) ceil(center_b - lmax[1]);
        int amin = (int) ceil(center_a - lmax[0]);

        int cmax = (int) floor(center_c + lmax[2]);
        int bmax = (int) floor(center_b + lmax[1]);
        int amax = (int) floor(center_a + lmax[0]);

        int c;
        for (c = cmin; c <= cmax; c++) {

            int ind_c = modulo(c * out_slice, out_size);
            double dc = center_c - c;
            double dz = grid_to_cartesian[8] * dc;
            double dz2 = dz * dz;
            double dy_c = grid_to_cartesian[5] * dc;
            double dx_c = grid_to_cartesian[2] * dc;

            int b;
            for (b = bmin; b <= bmax; b++) {
                int ind_cb = modulo(b * out_shape[2], out_slice) + ind_c;
                double db = center_b - b;
                double dy = dy_c + grid_to_cartesian[4] * db;
                double d2_zy = dz2 + dy * dy;
                double dx_cb = dx_c + grid_to_cartesian[1] * db;

                int a;
                for (a = amin; a <= amax; a++) {
                    double da = center_a - a;
                    double dx = dx_cb + grid_to_cartesian[0] * da;
                    double d2_zyx = d2_zy + dx * dx;
                    if (d2_zyx <= rmax2) {
                        double r = sqrt(d2_zyx);
                        int index =
                            radial_densities_ind + NEAREST_INT(r / rstep);
                        out[ind_cb + modulo(a, out_shape[2])] +=
                            q * radial_densities[index];
                    }
                }
            }
        }
    }

    // Clean up objects
    Py_DECREF(py_points);
    Py_DECREF(py_active);
    Py_DECREF(py_occupancies);
    Py_DECREF(py_lmax);
    Py_DECREF(py_radial_densities);
    Py_DECREF(py_grid_to_cartesian);
    PyArray_ResolveWritebackIfCopy(py_out);
    Py_DECREF(py_out);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    // Clean up objects
    Py_XDECREF(py_points);
    Py_XDECREF(py_radial_densities);
    Py_XDECREF(py_lmax);
    Py_XDECREF(py_active);
    Py_XDECREF(py_occupancies);
    Py_XDECREF(py_grid_to_cartesian);
    PyArray_DiscardWritebackIfCopy(py_out);
    Py_XDECREF(py_out);
    return NULL;
}


static PyMethodDef mymethods[] = {
    {"dilate_points", dilate_points, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef extensionmodule = {
    PyModuleDef_HEAD_INIT,
    "_extensions",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    mymethods
};


PyMODINIT_FUNC
PyInit__extensions(void)
{
    PyObject* m = PyModule_Create(&extensionmodule);
    if (!m) {
        return NULL;
    }
    import_array();
    return m;
};

