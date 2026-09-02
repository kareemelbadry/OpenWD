#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

/*
 * Formal solution for outward rays in a plane-parallel, pure-absorption
 * atmosphere.  Source functions are represented as linear in optical depth
 * inside each layer.  NumPy arrays are accepted through Python's buffer
 * protocol, so this extension does not depend on NumPy's C API.
 */

static int
is_double_buffer(const Py_buffer *view)
{
    return view->itemsize == (Py_ssize_t)sizeof(double) &&
           view->format != NULL &&
           (view->format[0] == 'd' ||
            (view->format[0] == '@' && view->format[1] == 'd') ||
            (view->format[0] == '=' && view->format[1] == 'd'));
}

static double
linear_source_coefficient(double delta)
{
    /* 1 - (1 + delta) exp(-delta), evaluated without cancellation. */
    if (delta < 1.0e-3) {
        return delta * delta *
               (0.5 + delta *
               (-1.0 / 3.0 + delta *
               (1.0 / 8.0 + delta *
               (-1.0 / 30.0 + delta / 144.0))));
    }
    return -expm1(-delta) - delta * exp(-delta);
}

static PyObject *
emergent_flux(PyObject *self, PyObject *args)
{
    PyObject *tau_obj = NULL;
    PyObject *source_obj = NULL;
    PyObject *mu_obj = NULL;
    PyObject *weight_obj = NULL;
    Py_buffer tau = {0};
    Py_buffer source = {0};
    Py_buffer mu = {0};
    Py_buffer weight = {0};
    PyObject *result = NULL;
    Py_ssize_t n_wave, n_depth, n_angle;
    Py_ssize_t wave, angle, depth;
    const double two_pi = 6.283185307179586476925286766559;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOO:emergent_flux",
                          &tau_obj, &source_obj, &mu_obj, &weight_obj)) {
        return NULL;
    }

    if (PyObject_GetBuffer(tau_obj, &tau,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0 ||
        PyObject_GetBuffer(source_obj, &source,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0 ||
        PyObject_GetBuffer(mu_obj, &mu,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0 ||
        PyObject_GetBuffer(weight_obj, &weight,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        goto cleanup;
    }

    if (!is_double_buffer(&tau) || !is_double_buffer(&source) ||
        !is_double_buffer(&mu) || !is_double_buffer(&weight)) {
        PyErr_SetString(PyExc_TypeError, "all inputs must have native float64 dtype");
        goto cleanup;
    }
    if (!PyBuffer_IsContiguous(&tau, 'C') ||
        !PyBuffer_IsContiguous(&source, 'C') ||
        !PyBuffer_IsContiguous(&mu, 'C') ||
        !PyBuffer_IsContiguous(&weight, 'C')) {
        PyErr_SetString(PyExc_ValueError, "all inputs must be C-contiguous");
        goto cleanup;
    }
    if (source.ndim != 2 || (tau.ndim != 1 && tau.ndim != 2) ||
        mu.ndim != 1 || weight.ndim != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "source must be 2D; tau 1D or 2D; angles and weights 1D");
        goto cleanup;
    }

    n_wave = source.shape[0];
    n_depth = source.shape[1];
    n_angle = mu.shape[0];
    if (n_depth < 2 || n_wave < 1 || n_angle < 1) {
        PyErr_SetString(PyExc_ValueError, "input arrays may not be empty");
        goto cleanup;
    }
    if (weight.shape[0] != n_angle ||
        (tau.ndim == 1 && tau.shape[0] != n_depth) ||
        (tau.ndim == 2 &&
         (tau.shape[0] != n_wave || tau.shape[1] != n_depth))) {
        PyErr_SetString(PyExc_ValueError, "input array shapes are inconsistent");
        goto cleanup;
    }

    result = PyList_New(n_wave);
    if (result == NULL) {
        goto cleanup;
    }

    for (wave = 0; wave < n_wave; ++wave) {
        const double *s = (const double *)source.buf + wave * n_depth;
        const double *t = (const double *)tau.buf;
        const double *mus = (const double *)mu.buf;
        const double *weights = (const double *)weight.buf;
        double flux = 0.0;

        if (tau.ndim == 2) {
            t += wave * n_depth;
        }
        for (angle = 0; angle < n_angle; ++angle) {
            const double ray_mu = mus[angle];
            double intensity = s[n_depth - 1];

            for (depth = n_depth - 2; depth >= 0; --depth) {
                const double delta = (t[depth + 1] - t[depth]) / ray_mu;
                const double attenuation = exp(-delta);
                const double constant_weight = -expm1(-delta);
                const double slope_weight =
                    linear_source_coefficient(delta) / delta;

                intensity = intensity * attenuation +
                            s[depth] * constant_weight +
                            (s[depth + 1] - s[depth]) * slope_weight;
            }
            {
                const double surface_delta = t[0] / ray_mu;
                const double surface_attenuation = exp(-surface_delta);
                intensity = intensity * surface_attenuation +
                            s[0] * (-expm1(-surface_delta));
            }
            flux += weights[angle] * ray_mu * intensity;
        }

        {
            PyObject *value = PyFloat_FromDouble(two_pi * flux);
            if (value == NULL) {
                Py_CLEAR(result);
                goto cleanup;
            }
            PyList_SET_ITEM(result, wave, value);
        }
    }

cleanup:
    if (tau.obj != NULL) {
        PyBuffer_Release(&tau);
    }
    if (source.obj != NULL) {
        PyBuffer_Release(&source);
    }
    if (mu.obj != NULL) {
        PyBuffer_Release(&mu);
    }
    if (weight.obj != NULL) {
        PyBuffer_Release(&weight);
    }
    return result;
}

/*
 * Convolve a profile represented by straight line segments on an irregular
 * wavelength grid with a normalized Lorentzian.  The segment integrals are
 * analytic.  Keeping the nested output/segment loops here avoids allocating
 * two (n_output, n_native - 1) temporary arrays for every atmospheric depth
 * and every He I line.
 */
static PyObject *
piecewise_linear_lorentz_convolution(PyObject *self, PyObject *args)
{
    PyObject *native_obj = NULL;
    PyObject *profile_obj = NULL;
    PyObject *offset_obj = NULL;
    Py_buffer native = {0};
    Py_buffer profile = {0};
    Py_buffer offset = {0};
    PyObject *result = NULL;
    Py_ssize_t n_native, n_output, output_index, segment;
    double gamma;
    const double pi = 3.1415926535897932384626433832795;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "OOOd:piecewise_linear_lorentz_convolution",
            &native_obj,
            &profile_obj,
            &offset_obj,
            &gamma)) {
        return NULL;
    }
    if (!isfinite(gamma) || gamma <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "gamma must be finite and positive");
        return NULL;
    }
    if (PyObject_GetBuffer(native_obj, &native,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0 ||
        PyObject_GetBuffer(profile_obj, &profile,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0 ||
        PyObject_GetBuffer(offset_obj, &offset,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        goto cleanup_lorentz;
    }
    if (!is_double_buffer(&native) || !is_double_buffer(&profile) ||
        !is_double_buffer(&offset)) {
        PyErr_SetString(PyExc_TypeError, "all arrays must have native float64 dtype");
        goto cleanup_lorentz;
    }
    if (!PyBuffer_IsContiguous(&native, 'C') ||
        !PyBuffer_IsContiguous(&profile, 'C') ||
        !PyBuffer_IsContiguous(&offset, 'C')) {
        PyErr_SetString(PyExc_ValueError, "all arrays must be C-contiguous");
        goto cleanup_lorentz;
    }
    if (native.ndim != 1 || profile.ndim != 1 || offset.ndim != 1) {
        PyErr_SetString(PyExc_ValueError, "all arrays must be one-dimensional");
        goto cleanup_lorentz;
    }
    n_native = native.shape[0];
    n_output = offset.shape[0];
    if (n_native < 2 || profile.shape[0] != n_native) {
        PyErr_SetString(PyExc_ValueError, "native and profile shapes are inconsistent");
        goto cleanup_lorentz;
    }

    result = PyList_New(n_output);
    if (result == NULL) {
        goto cleanup_lorentz;
    }
    {
        const double *x_native = (const double *)native.buf;
        const double *y_native = (const double *)profile.buf;
        const double *x_output = (const double *)offset.buf;
        const double gamma_squared = gamma * gamma;
        const double first_moment_scale = gamma / (2.0 * pi);

        for (segment = 0; segment < n_native; ++segment) {
            if (!isfinite(x_native[segment]) ||
                !isfinite(y_native[segment]) ||
                (segment > 0 &&
                 x_native[segment] <= x_native[segment - 1])) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "native grid must be finite and strictly increasing; profile must be finite"
                );
                Py_CLEAR(result);
                goto cleanup_lorentz;
            }
        }

        for (output_index = 0; output_index < n_output; ++output_index) {
            const double x = x_output[output_index];
            double value = 0.0;
            if (!isfinite(x)) {
                PyErr_SetString(PyExc_ValueError, "offset must be finite");
                Py_CLEAR(result);
                goto cleanup_lorentz;
            }
            for (segment = 0; segment < n_native - 1; ++segment) {
                const double spacing = x_native[segment + 1] - x_native[segment];
                const double left = x_native[segment] - x;
                const double right = x_native[segment + 1] - x;
                const double angle_integral =
                    (atan(right / gamma) - atan(left / gamma)) / pi;
                const double first_moment = first_moment_scale *
                    log((right * right + gamma_squared) /
                        (left * left + gamma_squared));
                const double slope =
                    (y_native[segment + 1] - y_native[segment]) / spacing;
                value += y_native[segment] * angle_integral +
                         slope * (first_moment +
                                  (x - x_native[segment]) * angle_integral);
            }
            {
                PyObject *item = PyFloat_FromDouble(value > 0.0 ? value : 0.0);
                if (item == NULL) {
                    Py_CLEAR(result);
                    goto cleanup_lorentz;
                }
                PyList_SET_ITEM(result, output_index, item);
            }
        }
    }

cleanup_lorentz:
    if (native.obj != NULL) {
        PyBuffer_Release(&native);
    }
    if (profile.obj != NULL) {
        PyBuffer_Release(&profile);
    }
    if (offset.obj != NULL) {
        PyBuffer_Release(&offset);
    }
    return result;
}

static Py_ssize_t
lower_bound_double(const double *values, Py_ssize_t size, double target)
{
    Py_ssize_t left = 0;
    Py_ssize_t right = size;
    while (left < right) {
        const Py_ssize_t middle = left + (right - left) / 2;
        if (values[middle] < target) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

static Py_ssize_t
upper_bound_double(const double *values, Py_ssize_t size, double target)
{
    Py_ssize_t left = 0;
    Py_ssize_t right = size;
    while (left < right) {
        const Py_ssize_t middle = left + (right - left) / 2;
        if (values[middle] <= target) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

static double
holtsmark_distribution(
    double beta,
    const double *quadrature_argument,
    const double *quadrature_weight,
    Py_ssize_t quadrature_size)
{
    const double pi = 3.1415926535897932384626433832795;
    double result;
    if (beta < 1.0e-3) {
        result = 4.0 / (3.0 * pi) * beta * beta;
    } else if (beta <= 8.0) {
        Py_ssize_t index;
        double integral = 0.0;
        for (index = 0; index < quadrature_size; ++index) {
            integral += quadrature_weight[index] *
                        sin(beta * quadrature_argument[index]);
        }
        result = 4.0 * beta / (3.0 * pi) * integral;
    } else {
        const double inverse = 1.0 / beta;
        result =
            1.496033551505373 * pow(inverse, 2.5) +
            7.639437268410976 * pow(inverse, 4.0) +
            21.598984399858832 * pow(inverse, 5.5) -
            447.50395803457565 * pow(inverse, 8.5) -
            3208.56365273261 * pow(inverse, 10.0) -
            12222.451853819328 * pow(inverse, 11.5);
    }
    return result > 0.0 ? result : 0.0;
}

/*
 * Accumulate ordinary LTE metal-line extinction.  Width and population
 * construction remain in Python; this kernel removes the expensive
 * line/depth/profile Python loop used by cool metal-dominated atmospheres.
 */
static PyObject *
accumulate_lte_metal_line_profiles(PyObject *self, PyObject *args)
{
    PyObject *objects[8] = {NULL};
    Py_buffer views[8] = {{0}};
    Py_ssize_t n_wave, n_depth, n_line, line, depth, wave;
    int index;
    const double pi = 3.1415926535897932384626433832795;
    const double light_speed = 2.99792458e10;
    const double log_two = 0.69314718055994530941723212145818;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOO:accumulate_lte_metal_line_profiles",
            &objects[0], &objects[1], &objects[2], &objects[3],
            &objects[4], &objects[5], &objects[6], &objects[7])) {
        return NULL;
    }
    for (index = 0; index < 8; ++index) {
        const int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES |
                          (index == 7 ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(objects[index], &views[index], flags) < 0) {
            goto cleanup_lte_metal;
        }
        if (!is_double_buffer(&views[index])) {
            PyErr_SetString(PyExc_TypeError,
                            "all LTE metal-profile arrays must have native float64 dtype");
            goto cleanup_lte_metal;
        }
        if (!PyBuffer_IsContiguous(&views[index], 'C')) {
            PyErr_SetString(PyExc_ValueError,
                            "all LTE metal-profile arrays must be C-contiguous");
            goto cleanup_lte_metal;
        }
    }
    if (views[0].ndim != 1 || views[1].ndim != 1 ||
        views[2].ndim != 1 || views[3].ndim != 2 ||
        views[4].ndim != 2 || views[5].ndim != 1 ||
        views[6].ndim != 2 || views[7].ndim != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "LTE metal-profile arrays have invalid ranks");
        goto cleanup_lte_metal;
    }
    n_wave = views[0].shape[0];
    n_line = views[1].shape[0];
    n_depth = views[7].shape[1];
    if (n_wave < 1 || n_line < 1 || n_depth < 1 ||
        views[2].shape[0] != n_line ||
        views[3].shape[0] != n_line || views[3].shape[1] != n_depth ||
        views[4].shape[0] != n_line || views[4].shape[1] != n_depth ||
        views[5].shape[0] != n_line ||
        views[6].shape[0] != n_line || views[6].shape[1] != n_depth ||
        views[7].shape[0] != n_wave) {
        PyErr_SetString(PyExc_ValueError,
                        "LTE metal-profile array shapes are inconsistent");
        goto cleanup_lte_metal;
    }

    {
        const double *wavelength = (const double *)views[0].buf;
        const double *center = (const double *)views[1].buf;
        const double *strength = (const double *)views[2].buf;
        const double *gaussian_sigma = (const double *)views[3].buf;
        const double *lorentz_hwhm = (const double *)views[4].buf;
        const double *minimum_half_window = (const double *)views[5].buf;
        const double *population_scale = (const double *)views[6].buf;
        double *absorption = (double *)views[7].buf;

        Py_BEGIN_ALLOW_THREADS
        for (line = 0; line < n_line; ++line) {
            const double line_center = center[line];
            const double center_cm = line_center * 1.0e-8;
            const double frequency_conversion =
                1.0e8 * center_cm * center_cm / light_speed;
            for (depth = 0; depth < n_depth; ++depth) {
                const Py_ssize_t line_depth = line * n_depth + depth;
                const double sigma = gaussian_sigma[line_depth];
                const double gamma = lorentz_hwhm[line_depth];
                double half_window = 0.25;
                double gaussian_fwhm, lorentz_fwhm, width, ratio, mixing;
                double profile_scale;
                Py_ssize_t start, stop;

                if (minimum_half_window[line] > half_window) {
                    half_window = minimum_half_window[line];
                }
                if (10.0 * sigma > half_window) {
                    half_window = 10.0 * sigma;
                }
                if (100.0 * gamma > half_window) {
                    half_window = 100.0 * gamma;
                }
                start = lower_bound_double(
                    wavelength, n_wave, line_center - half_window);
                stop = upper_bound_double(
                    wavelength, n_wave, line_center + half_window);
                if (stop <= start) {
                    continue;
                }

                gaussian_fwhm =
                    2.0 * sqrt(2.0 * log_two) * fmax(sigma, 1.0e-12);
                lorentz_fwhm = 2.0 * fmax(gamma, 0.0);
                width = pow(
                    pow(gaussian_fwhm, 5.0) +
                    2.69269 * pow(gaussian_fwhm, 4.0) * lorentz_fwhm +
                    2.42843 * pow(gaussian_fwhm, 3.0) *
                        lorentz_fwhm * lorentz_fwhm +
                    4.47163 * gaussian_fwhm * gaussian_fwhm *
                        pow(lorentz_fwhm, 3.0) +
                    0.07842 * gaussian_fwhm * pow(lorentz_fwhm, 4.0) +
                    pow(lorentz_fwhm, 5.0),
                    0.2);
                ratio = lorentz_fwhm / width;
                mixing = 1.36603 * ratio - 0.47719 * ratio * ratio +
                         0.11116 * ratio * ratio * ratio;
                if (mixing < 0.0) {
                    mixing = 0.0;
                } else if (mixing > 1.0) {
                    mixing = 1.0;
                }
                profile_scale = strength[line] * frequency_conversion *
                    population_scale[line_depth];

                for (wave = start; wave < stop; ++wave) {
                    const double offset = wavelength[wave] - line_center;
                    const double normalized_offset = offset / width;
                    const double gaussian =
                        2.0 * sqrt(log_two) / (sqrt(pi) * width) *
                        exp(-4.0 * log_two * normalized_offset * normalized_offset);
                    const double lorentz =
                        2.0 / (pi * width) /
                        (1.0 + 4.0 * normalized_offset * normalized_offset);
                    const double profile =
                        mixing * lorentz + (1.0 - mixing) * gaussian;
                    absorption[wave * n_depth + depth] +=
                        profile * profile_scale;
                }
            }
        }
        Py_END_ALLOW_THREADS
    }

    for (index = 0; index < 8; ++index) {
        PyBuffer_Release(&views[index]);
    }
    Py_RETURN_NONE;

cleanup_lte_metal:
    for (index = 0; index < 8; ++index) {
        if (views[index].obj != NULL) {
            PyBuffer_Release(&views[index]);
        }
    }
    return NULL;
}

/*
 * Accumulate a batch of ordinary metal-line extinction and emissivity.
 * Atomic-data interpretation and width construction stay in Python; this
 * kernel performs only the element-independent profile arithmetic.  Keeping
 * the output arrays caller-owned avoids constructing enormous Python lists.
 */
static PyObject *
accumulate_metal_line_profiles(PyObject *self, PyObject *args)
{
    PyObject *objects[17] = {NULL};
    Py_buffer views[17] = {{0}};
    Py_ssize_t n_wave, n_depth, n_line, line, depth, wave;
    int retain_inverted_emissivity;
    int index;
    const double pi = 3.1415926535897932384626433832795;
    const double light_speed = 2.99792458e10;
    const double log_two = 0.69314718055994530941723212145818;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOOOOOOOp:accumulate_metal_line_profiles",
            &objects[0], &objects[1], &objects[2], &objects[3],
            &objects[4], &objects[5], &objects[6], &objects[7],
            &objects[8], &objects[9], &objects[10], &objects[11],
            &objects[12], &objects[13], &objects[14], &objects[15],
            &objects[16],
            &retain_inverted_emissivity)) {
        return NULL;
    }
    for (index = 0; index < 17; ++index) {
        const int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES |
                          (index >= 15 ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(objects[index], &views[index], flags) < 0) {
            goto cleanup_metal;
        }
        if (!is_double_buffer(&views[index])) {
            PyErr_SetString(PyExc_TypeError,
                            "all metal-profile arrays must have native float64 dtype");
            goto cleanup_metal;
        }
        if (!PyBuffer_IsContiguous(&views[index], 'C')) {
            PyErr_SetString(PyExc_ValueError,
                            "all metal-profile arrays must be C-contiguous");
            goto cleanup_metal;
        }
    }

    if (views[0].ndim != 1 || views[1].ndim != 2 ||
        views[2].ndim != 1 || views[3].ndim != 1 ||
        views[6].ndim != 1 || views[13].ndim != 1 ||
        views[14].ndim != 1 || views[15].ndim != 2 ||
        views[16].ndim != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "metal-profile wavelength, line and quadrature arrays have invalid ranks");
        goto cleanup_metal;
    }
    n_wave = views[0].shape[0];
    n_depth = views[1].shape[1];
    n_line = views[2].shape[0];
    if (n_wave < 1 || n_depth < 1 || n_line < 1 ||
        views[1].shape[0] != n_wave || views[3].shape[0] != n_line ||
        views[6].shape[0] != n_line ||
        views[13].shape[0] != views[14].shape[0] ||
        views[13].shape[0] < 1 ||
        views[15].shape[0] != n_wave || views[15].shape[1] != n_depth ||
        views[16].shape[0] != n_wave || views[16].shape[1] != n_depth) {
        PyErr_SetString(PyExc_ValueError,
                        "metal-profile array shapes are inconsistent");
        goto cleanup_metal;
    }
    for (index = 4; index <= 12; ++index) {
        if (index == 6) {
            continue;
        }
        if (views[index].ndim != 2 ||
            views[index].shape[0] != n_line ||
            views[index].shape[1] != n_depth) {
            PyErr_SetString(PyExc_ValueError,
                            "per-line metal-profile arrays must have shape (line, depth)");
            goto cleanup_metal;
        }
    }

    {
        const double *wavelength = (const double *)views[0].buf;
        const double *planck = (const double *)views[1].buf;
        const double *center = (const double *)views[2].buf;
        const double *strength = (const double *)views[3].buf;
        const double *gaussian_sigma = (const double *)views[4].buf;
        const double *lorentz_hwhm = (const double *)views[5].buf;
        const double *minimum_half_window = (const double *)views[6].buf;
        const double *static_scale = (const double *)views[7].buf;
        const double *static_amplitude = (const double *)views[8].buf;
        const double *population_scale = (const double *)views[9].buf;
        const double *lower_departure = (const double *)views[10].buf;
        const double *upper_departure = (const double *)views[11].buf;
        const double *exponential = (const double *)views[12].buf;
        const double *quadrature_argument = (const double *)views[13].buf;
        const double *quadrature_weight = (const double *)views[14].buf;
        double *absorption = (double *)views[15].buf;
        double *emissivity = (double *)views[16].buf;
        const Py_ssize_t quadrature_size = views[13].shape[0];

        Py_BEGIN_ALLOW_THREADS
        for (line = 0; line < n_line; ++line) {
            const double line_center = center[line];
            const double center_cm = line_center * 1.0e-8;
            const double frequency_conversion =
                1.0e8 * center_cm * center_cm / light_speed;
            const double center_frequency = light_speed / center_cm;
            for (depth = 0; depth < n_depth; ++depth) {
                const Py_ssize_t line_depth = line * n_depth + depth;
                const double sigma = gaussian_sigma[line_depth];
                const double gamma = lorentz_hwhm[line_depth];
                const double field_scale = static_scale[line_depth];
                const double static_half_window =
                    30.0 * field_scale * center_cm * center_cm /
                    light_speed * 1.0e8;
                double half_window = 0.25;
                double gaussian_fwhm, lorentz_fwhm, width, ratio, mixing;
                Py_ssize_t start, stop;
                double net_departure, stimulated, line_absorption_scale;
                double line_emissivity_scale;

                if (minimum_half_window[line] > half_window) {
                    half_window = minimum_half_window[line];
                }
                if (10.0 * sigma > half_window) {
                    half_window = 10.0 * sigma;
                }
                if (100.0 * gamma > half_window) {
                    half_window = 100.0 * gamma;
                }
                if (static_half_window > half_window) {
                    half_window = static_half_window;
                }
                start = lower_bound_double(
                    wavelength, n_wave, line_center - half_window);
                stop = upper_bound_double(
                    wavelength, n_wave, line_center + half_window);
                if (stop <= start) {
                    continue;
                }

                gaussian_fwhm =
                    2.0 * sqrt(2.0 * log_two) * fmax(sigma, 1.0e-12);
                lorentz_fwhm = 2.0 * fmax(gamma, 0.0);
                width = pow(
                    pow(gaussian_fwhm, 5.0) +
                    2.69269 * pow(gaussian_fwhm, 4.0) * lorentz_fwhm +
                    2.42843 * pow(gaussian_fwhm, 3.0) *
                        lorentz_fwhm * lorentz_fwhm +
                    4.47163 * gaussian_fwhm * gaussian_fwhm *
                        pow(lorentz_fwhm, 3.0) +
                    0.07842 * gaussian_fwhm * pow(lorentz_fwhm, 4.0) +
                    pow(lorentz_fwhm, 5.0),
                    0.2);
                ratio = lorentz_fwhm / width;
                mixing = 1.36603 * ratio - 0.47719 * ratio * ratio +
                         0.11116 * ratio * ratio * ratio;
                if (mixing < 0.0) {
                    mixing = 0.0;
                } else if (mixing > 1.0) {
                    mixing = 1.0;
                }

                net_departure = lower_departure[line_depth] -
                    upper_departure[line_depth] * exponential[line_depth];
                stimulated = 1.0 - exponential[line_depth];
                line_absorption_scale = population_scale[line_depth] *
                    net_departure;
                line_emissivity_scale = population_scale[line_depth] *
                    stimulated * upper_departure[line_depth];
                if (net_departure <= 0.0 && !retain_inverted_emissivity) {
                    continue;
                }

                for (wave = start; wave < stop; ++wave) {
                    const double offset = wavelength[wave] - line_center;
                    const double normalized_offset = offset / width;
                    const double gaussian =
                        2.0 * sqrt(log_two) / (sqrt(pi) * width) *
                        exp(-4.0 * log_two * normalized_offset * normalized_offset);
                    const double lorentz =
                        2.0 / (pi * width) /
                        (1.0 + 4.0 * normalized_offset * normalized_offset);
                    const double profile =
                        mixing * lorentz + (1.0 - mixing) * gaussian;
                    double cross_section =
                        strength[line] * profile * frequency_conversion;
                    const Py_ssize_t wave_depth = wave * n_depth + depth;

                    if (field_scale > 0.0) {
                        const double frequency =
                            light_speed / (wavelength[wave] * 1.0e-8);
                        const double beta =
                            fabs(frequency - center_frequency) / field_scale;
                        if (beta <= 30.0) {
                            const double static_cross_section =
                                0.5 * static_amplitude[line_depth] *
                                holtsmark_distribution(
                                    beta,
                                    quadrature_argument,
                                    quadrature_weight,
                                    quadrature_size);
                            if (static_cross_section > cross_section) {
                                cross_section = static_cross_section;
                            }
                        }
                    }
                    if (net_departure > 0.0) {
                        absorption[wave_depth] +=
                            cross_section * line_absorption_scale;
                    }
                    emissivity[wave_depth] +=
                        cross_section * line_emissivity_scale *
                        planck[wave_depth];
                }
            }
        }
        Py_END_ALLOW_THREADS
    }

    for (index = 0; index < 16; ++index) {
        PyBuffer_Release(&views[index]);
    }
    Py_RETURN_NONE;

cleanup_metal:
    for (index = 0; index < 16; ++index) {
        if (views[index].obj != NULL) {
            PyBuffer_Release(&views[index]);
        }
    }
    return NULL;
}

static double
linear_interpolate(
    const double *wavelength,
    const double *values,
    Py_ssize_t n_wave,
    Py_ssize_t n_depth,
    Py_ssize_t depth,
    double target)
{
    Py_ssize_t right;
    if (target <= wavelength[0]) {
        return values[depth];
    }
    if (target >= wavelength[n_wave - 1]) {
        return values[(n_wave - 1) * n_depth + depth];
    }
    right = upper_bound_double(wavelength, n_wave, target);
    {
        const Py_ssize_t left = right - 1;
        const double fraction =
            (target - wavelength[left]) /
            (wavelength[right] - wavelength[left]);
        return values[left * n_depth + depth] * (1.0 - fraction) +
               values[right * n_depth + depth] * fraction;
    }
}

/* Profile-weighted radiation fields used to construct bound-bound rates. */
static PyObject *
metal_line_mean_intensity(PyObject *self, PyObject *args)
{
    PyObject *objects[8] = {NULL};
    Py_buffer views[8] = {{0}};
    Py_ssize_t n_wave, n_depth, n_line, line, depth;
    int include_lambda_diagonal;
    int index;
    const double pi = 3.1415926535897932384626433832795;
    const double log_two = 0.69314718055994530941723212145818;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOp:metal_line_mean_intensity",
            &objects[0], &objects[1], &objects[2], &objects[3],
            &objects[4], &objects[5], &objects[6], &objects[7],
            &include_lambda_diagonal)) {
        return NULL;
    }
    for (index = 0; index < 8; ++index) {
        const int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES |
                          (index >= 6 ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(objects[index], &views[index], flags) < 0) {
            goto cleanup_line_mean;
        }
        if (!is_double_buffer(&views[index])) {
            PyErr_SetString(PyExc_TypeError,
                            "all line-mean arrays must have native float64 dtype");
            goto cleanup_line_mean;
        }
        if (!PyBuffer_IsContiguous(&views[index], 'C')) {
            PyErr_SetString(PyExc_ValueError,
                            "all line-mean arrays must be C-contiguous");
            goto cleanup_line_mean;
        }
    }
    if (views[0].ndim != 1 || views[1].ndim != 2 || views[2].ndim != 2 ||
        views[3].ndim != 1 || views[4].ndim != 2 || views[5].ndim != 2 ||
        views[6].ndim != 2 || views[7].ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "line-mean arrays have invalid ranks");
        goto cleanup_line_mean;
    }
    n_wave = views[0].shape[0];
    n_depth = views[1].shape[1];
    n_line = views[3].shape[0];
    if (n_wave < 1 || n_depth < 1 ||
        views[1].shape[0] != n_wave ||
        views[2].shape[0] != n_wave || views[2].shape[1] != n_depth ||
        views[4].shape[0] != n_line || views[4].shape[1] != n_depth ||
        views[5].shape[0] != n_line || views[5].shape[1] != n_depth ||
        views[6].shape[0] != n_line || views[6].shape[1] != n_depth ||
        views[7].shape[0] != n_line || views[7].shape[1] != n_depth) {
        PyErr_SetString(PyExc_ValueError, "line-mean array shapes are inconsistent");
        goto cleanup_line_mean;
    }

    {
        const double *wavelength = (const double *)views[0].buf;
        const double *intensity = (const double *)views[1].buf;
        const double *lambda_diagonal = (const double *)views[2].buf;
        const double *center = (const double *)views[3].buf;
        const double *gaussian_sigma = (const double *)views[4].buf;
        const double *lorentz_hwhm = (const double *)views[5].buf;
        double *mean_intensity = (double *)views[6].buf;
        double *mean_lambda = (double *)views[7].buf;

        Py_BEGIN_ALLOW_THREADS
        for (line = 0; line < n_line; ++line) {
            for (depth = 0; depth < n_depth; ++depth) {
                const Py_ssize_t line_depth = line * n_depth + depth;
                const double sigma = gaussian_sigma[line_depth];
                const double gamma = lorentz_hwhm[line_depth];
                const double half_width = fmax(7.0 * sigma, 100.0 * gamma);
                const Py_ssize_t start = lower_bound_double(
                    wavelength, n_wave, center[line] - half_width);
                const Py_ssize_t stop = upper_bound_double(
                    wavelength, n_wave, center[line] + half_width);
                if (stop - start >= 3) {
                    const double gaussian_fwhm =
                        2.0 * sqrt(2.0 * log_two) * fmax(sigma, 1.0e-12);
                    const double lorentz_fwhm = 2.0 * fmax(gamma, 0.0);
                    const double width = pow(
                        pow(gaussian_fwhm, 5.0) +
                        2.69269 * pow(gaussian_fwhm, 4.0) * lorentz_fwhm +
                        2.42843 * pow(gaussian_fwhm, 3.0) *
                            lorentz_fwhm * lorentz_fwhm +
                        4.47163 * gaussian_fwhm * gaussian_fwhm *
                            pow(lorentz_fwhm, 3.0) +
                        0.07842 * gaussian_fwhm * pow(lorentz_fwhm, 4.0) +
                        pow(lorentz_fwhm, 5.0),
                        0.2);
                    double ratio = lorentz_fwhm / width;
                    double mixing = 1.36603 * ratio - 0.47719 * ratio * ratio +
                                    0.11116 * ratio * ratio * ratio;
                    Py_ssize_t wave;
                    double previous_profile, normalization = 0.0;
                    double intensity_integral = 0.0;
                    double lambda_integral = 0.0;
                    if (mixing < 0.0) {
                        mixing = 0.0;
                    } else if (mixing > 1.0) {
                        mixing = 1.0;
                    }
#define PROFILE_AT(WAVE_INDEX) ( \
    mixing * (2.0 / (pi * width) / \
        (1.0 + 4.0 * pow((wavelength[(WAVE_INDEX)] - center[line]) / width, 2.0))) + \
    (1.0 - mixing) * (2.0 * sqrt(log_two) / (sqrt(pi) * width) * \
        exp(-4.0 * log_two * pow((wavelength[(WAVE_INDEX)] - center[line]) / width, 2.0))) )
                    previous_profile = PROFILE_AT(start);
                    for (wave = start + 1; wave < stop; ++wave) {
                        const double profile = PROFILE_AT(wave);
                        const double spacing = wavelength[wave] - wavelength[wave - 1];
                        const Py_ssize_t current = wave * n_depth + depth;
                        const Py_ssize_t previous = (wave - 1) * n_depth + depth;
                        normalization += 0.5 * spacing *
                            (previous_profile + profile);
                        intensity_integral += 0.5 * spacing *
                            (intensity[previous] * previous_profile +
                             intensity[current] * profile);
                        if (include_lambda_diagonal) {
                            lambda_integral += 0.5 * spacing *
                                (lambda_diagonal[previous] * previous_profile +
                                 lambda_diagonal[current] * profile);
                        }
                        previous_profile = profile;
                    }
#undef PROFILE_AT
                    mean_intensity[line_depth] =
                        intensity_integral / normalization;
                    mean_lambda[line_depth] = include_lambda_diagonal ?
                        lambda_integral / normalization : 0.0;
                } else {
                    mean_intensity[line_depth] = linear_interpolate(
                        wavelength, intensity, n_wave, n_depth, depth,
                        center[line]);
                    mean_lambda[line_depth] = include_lambda_diagonal ?
                        linear_interpolate(
                            wavelength, lambda_diagonal, n_wave, n_depth,
                            depth, center[line]) : 0.0;
                }
            }
        }
        Py_END_ALLOW_THREADS
    }

    for (index = 0; index < 8; ++index) {
        PyBuffer_Release(&views[index]);
    }
    Py_RETURN_NONE;

cleanup_line_mean:
    for (index = 0; index < 8; ++index) {
        if (views[index].obj != NULL) {
            PyBuffer_Release(&views[index]);
        }
    }
    return NULL;
}

/* Bound-free radiative rates for one or more level cross sections. */
static PyObject *
photoionization_rates(PyObject *self, PyObject *args)
{
    PyObject *objects[4] = {NULL};
    Py_buffer views[4] = {{0}};
    Py_ssize_t n_wave, n_depth, n_level, level, depth, wave;
    int index;
    const double rate_prefactor =
        12.566370614359172953850573533118 * 1.0e-8 /
        (6.62607015e-27 * 2.99792458e10);

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "OOOO:photoionization_rates",
            &objects[0], &objects[1], &objects[2], &objects[3])) {
        return NULL;
    }
    for (index = 0; index < 4; ++index) {
        const int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES |
                          (index == 3 ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(objects[index], &views[index], flags) < 0) {
            goto cleanup_photoionization;
        }
        if (!is_double_buffer(&views[index])) {
            PyErr_SetString(PyExc_TypeError,
                            "all photoionization arrays must have native float64 dtype");
            goto cleanup_photoionization;
        }
        if (!PyBuffer_IsContiguous(&views[index], 'C')) {
            PyErr_SetString(PyExc_ValueError,
                            "all photoionization arrays must be C-contiguous");
            goto cleanup_photoionization;
        }
    }
    if (views[0].ndim != 1 || views[1].ndim != 2 ||
        views[2].ndim != 2 || views[3].ndim != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "photoionization arrays have invalid ranks");
        goto cleanup_photoionization;
    }
    n_wave = views[0].shape[0];
    n_depth = views[1].shape[1];
    n_level = views[2].shape[0];
    if (n_wave < 2 || n_depth < 1 ||
        views[1].shape[0] != n_wave || views[2].shape[1] != n_wave ||
        views[3].shape[0] != n_level || views[3].shape[1] != n_depth) {
        PyErr_SetString(PyExc_ValueError,
                        "photoionization array shapes are inconsistent");
        goto cleanup_photoionization;
    }

    {
        const double *wavelength = (const double *)views[0].buf;
        const double *intensity = (const double *)views[1].buf;
        const double *cross_section = (const double *)views[2].buf;
        double *rate = (double *)views[3].buf;
        Py_BEGIN_ALLOW_THREADS
        for (level = 0; level < n_level; ++level) {
            for (depth = 0; depth < n_depth; ++depth) {
                double integral = 0.0;
                double previous =
                    intensity[depth] * cross_section[level * n_wave] *
                    wavelength[0] * rate_prefactor;
                for (wave = 1; wave < n_wave; ++wave) {
                    const double current =
                        intensity[wave * n_depth + depth] *
                        cross_section[level * n_wave + wave] *
                        wavelength[wave] * rate_prefactor;
                    integral += 0.5 * (previous + current) *
                                (wavelength[wave] - wavelength[wave - 1]);
                    previous = current;
                }
                rate[level * n_depth + depth] = integral;
            }
        }
        Py_END_ALLOW_THREADS
    }

    for (index = 0; index < 4; ++index) {
        PyBuffer_Release(&views[index]);
    }
    Py_RETURN_NONE;

cleanup_photoionization:
    for (index = 0; index < 4; ++index) {
        if (views[index].obj != NULL) {
            PyBuffer_Release(&views[index]);
        }
    }
    return NULL;
}

/*
 * Exact response of the wavelength-integrated conservative Feautrier
 * interface flux to local source-function and mass-opacity perturbations.
 *
 * This is the compiled counterpart of
 * integrated_feautrier_interface_state_response() in radiative_transfer.py.
 * The algebra is deliberately kept identical to the Python reference: the
 * opacity response moves every deeper optical-depth node, the complete
 * tridiagonal system is differentiated, and the optical-depth denominator in
 * the interface flux is differentiated as well.  Moving the O(N_lambda *
 * N_angle * N_depth^2) tangent sweeps here changes no atmosphere physics; it
 * only removes tens of thousands of small Python/NumPy operations from every
 * full structure-Jacobian refresh.
 */
static PyObject *
integrated_feautrier_interface_state_response(PyObject *self, PyObject *args)
{
    PyObject *objects[8] = {NULL};
    Py_buffer views[8] = {{0}};
    PyObject *result = NULL;
    double *integrated = NULL;
    double *expanded_tau = NULL;
    double *expanded_source = NULL;
    double *expanded_tau_response = NULL;
    double *lower = NULL;
    double *diagonal = NULL;
    double *upper = NULL;
    double *lower_response = NULL;
    double *diagonal_response = NULL;
    double *upper_response = NULL;
    double *right_hand_side = NULL;
    double *multipliers = NULL;
    double *symmetric_intensity = NULL;
    double *tangent_rhs = NULL;
    double *symmetric_response = NULL;
    Py_ssize_t n_wave, n_depth, n_node, n_angle;
    Py_ssize_t wave, angle, depth, state, index;
    const double four_pi = 12.566370614359172953850573533118;

    (void)self;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOO:integrated_feautrier_interface_state_response",
            &objects[0], &objects[1], &objects[2], &objects[3],
            &objects[4], &objects[5], &objects[6], &objects[7])) {
        return NULL;
    }
    for (index = 0; index < 8; ++index) {
        if (PyObject_GetBuffer(
                objects[index], &views[index],
                PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
            goto cleanup_feautrier_response;
        }
        if (!is_double_buffer(&views[index]) ||
            !PyBuffer_IsContiguous(&views[index], 'C')) {
            PyErr_SetString(
                PyExc_TypeError,
                "all Feautrier response inputs must be C-contiguous native float64 arrays"
            );
            goto cleanup_feautrier_response;
        }
    }
    if (views[0].ndim != 2 || views[1].ndim != 2 ||
        views[2].ndim != 2 || views[3].ndim != 2 ||
        views[4].ndim != 1 || views[5].ndim != 1 ||
        views[6].ndim != 1 || views[7].ndim != 1) {
        PyErr_SetString(PyExc_ValueError, "Feautrier response input ranks are inconsistent");
        goto cleanup_feautrier_response;
    }
    n_wave = views[0].shape[0];
    n_depth = views[0].shape[1];
    n_node = n_depth + 1;
    n_angle = views[5].shape[0];
    if (n_wave < 2 || n_depth < 2 || n_angle < 1 ||
        views[1].shape[0] != n_wave || views[1].shape[1] != n_depth ||
        views[2].shape[0] != n_wave || views[2].shape[1] != n_depth ||
        views[3].shape[0] != n_wave || views[3].shape[1] != n_depth ||
        views[4].shape[0] != n_depth ||
        views[6].shape[0] != n_angle || views[7].shape[0] != n_wave) {
        PyErr_SetString(PyExc_ValueError, "Feautrier response input shapes are inconsistent");
        goto cleanup_feautrier_response;
    }

#define ALLOCATE_RESPONSE_ARRAY(name, count)                                      \
    do {                                                                           \
        name = (double *)PyMem_Malloc((size_t)(count) * sizeof(double));           \
        if (name == NULL) {                                                         \
            PyErr_NoMemory();                                                       \
            goto cleanup_feautrier_response;                                        \
        }                                                                           \
    } while (0)

    ALLOCATE_RESPONSE_ARRAY(integrated, n_depth * n_depth);
    ALLOCATE_RESPONSE_ARRAY(expanded_tau, n_node);
    ALLOCATE_RESPONSE_ARRAY(expanded_source, n_node);
    ALLOCATE_RESPONSE_ARRAY(expanded_tau_response, n_node * n_depth);
    ALLOCATE_RESPONSE_ARRAY(lower, n_node);
    ALLOCATE_RESPONSE_ARRAY(diagonal, n_node);
    ALLOCATE_RESPONSE_ARRAY(upper, n_node);
    ALLOCATE_RESPONSE_ARRAY(lower_response, n_node * n_depth);
    ALLOCATE_RESPONSE_ARRAY(diagonal_response, n_node * n_depth);
    ALLOCATE_RESPONSE_ARRAY(upper_response, n_node * n_depth);
    ALLOCATE_RESPONSE_ARRAY(right_hand_side, n_node);
    ALLOCATE_RESPONSE_ARRAY(multipliers, n_node);
    ALLOCATE_RESPONSE_ARRAY(symmetric_intensity, n_node);
    ALLOCATE_RESPONSE_ARRAY(tangent_rhs, n_node * n_depth);
    ALLOCATE_RESPONSE_ARRAY(symmetric_response, n_node * n_depth);
#undef ALLOCATE_RESPONSE_ARRAY

    for (index = 0; index < n_depth * n_depth; ++index) {
        integrated[index] = 0.0;
    }

    {
        const double *tau = (const double *)views[0].buf;
        const double *source = (const double *)views[1].buf;
        const double *source_response = (const double *)views[2].buf;
        const double *opacity_response = (const double *)views[3].buf;
        const double *mass = (const double *)views[4].buf;
        const double *mu = (const double *)views[5].buf;
        const double *angle_weight = (const double *)views[6].buf;
        const double *wavelength_weight = (const double *)views[7].buf;

        Py_BEGIN_ALLOW_THREADS
        for (wave = 0; wave < n_wave; ++wave) {
            const double *local_tau = tau + wave * n_depth;
            const double *local_source = source + wave * n_depth;
            const double *local_source_response =
                source_response + wave * n_depth;
            const double *local_opacity_response =
                opacity_response + wave * n_depth;

            expanded_tau[0] = 0.0;
            expanded_source[0] = local_source[0];
            for (depth = 0; depth < n_depth; ++depth) {
                expanded_tau[depth + 1] = local_tau[depth];
                expanded_source[depth + 1] = local_source[depth];
            }
            for (index = 0; index < n_node * n_depth; ++index) {
                expanded_tau_response[index] = 0.0;
            }
            expanded_tau_response[n_depth] =
                local_opacity_response[0] * mass[0];
            for (depth = 1; depth < n_depth; ++depth) {
                const Py_ssize_t row = (depth + 1) * n_depth;
                const Py_ssize_t previous_row = depth * n_depth;
                const double mass_step = mass[depth] - mass[depth - 1];
                for (state = 0; state < n_depth; ++state) {
                    expanded_tau_response[row + state] =
                        expanded_tau_response[previous_row + state];
                }
                expanded_tau_response[row + depth - 1] +=
                    0.5 * mass_step * local_opacity_response[depth - 1];
                expanded_tau_response[row + depth] +=
                    0.5 * mass_step * local_opacity_response[depth];
            }

            for (angle = 0; angle < n_angle; ++angle) {
                const double ray_mu = mu[angle];
                const double ray_scale = four_pi * angle_weight[angle]
                                         * ray_mu * ray_mu;
                const double first_step = expanded_tau[1];

                for (depth = 0; depth < n_node; ++depth) {
                    lower[depth] = 0.0;
                    diagonal[depth] = 0.0;
                    upper[depth] = 0.0;
                    multipliers[depth] = 0.0;
                }
                lower[0] = 0.0;
                diagonal[0] = 1.0 + ray_mu / first_step;
                upper[0] = -ray_mu / first_step;
                for (depth = 1; depth < n_depth; ++depth) {
                    const double previous =
                        expanded_tau[depth] - expanded_tau[depth - 1];
                    const double following =
                        expanded_tau[depth + 1] - expanded_tau[depth];
                    lower[depth] = -2.0 * ray_mu * ray_mu /
                        (previous * (previous + following));
                    upper[depth] = -2.0 * ray_mu * ray_mu /
                        (following * (previous + following));
                    diagonal[depth] = 1.0 - lower[depth] - upper[depth];
                }
                diagonal[n_depth] = 1.0;

                for (index = 0; index < n_node * n_depth; ++index) {
                    lower_response[index] = 0.0;
                    diagonal_response[index] = 0.0;
                    upper_response[index] = 0.0;
                }
                for (state = 0; state < n_depth; ++state) {
                    const double first_response =
                        expanded_tau_response[n_depth + state];
                    diagonal_response[state] =
                        -ray_mu * first_response / (first_step * first_step);
                    upper_response[state] = -diagonal_response[state];
                }
                for (depth = 1; depth < n_depth; ++depth) {
                    const double previous =
                        expanded_tau[depth] - expanded_tau[depth - 1];
                    const double following =
                        expanded_tau[depth + 1] - expanded_tau[depth];
                    const double combined = previous + following;
                    const Py_ssize_t row = depth * n_depth;
                    const Py_ssize_t previous_row = (depth - 1) * n_depth;
                    const Py_ssize_t following_row = (depth + 1) * n_depth;
                    for (state = 0; state < n_depth; ++state) {
                        const double previous_response =
                            expanded_tau_response[row + state]
                            - expanded_tau_response[previous_row + state];
                        const double following_response =
                            expanded_tau_response[following_row + state]
                            - expanded_tau_response[row + state];
                        const double combined_response =
                            previous_response + following_response;
                        lower_response[row + state] = lower[depth] *
                            (-previous_response / previous
                             - combined_response / combined);
                        upper_response[row + state] = upper[depth] *
                            (-following_response / following
                             - combined_response / combined);
                        diagonal_response[row + state] =
                            -lower_response[row + state]
                            -upper_response[row + state];
                    }
                }

                right_hand_side[0] = 0.0;
                for (depth = 1; depth < n_depth; ++depth) {
                    right_hand_side[depth] = expanded_source[depth];
                }
                right_hand_side[n_depth] = expanded_source[n_depth];
                for (depth = 1; depth < n_node; ++depth) {
                    const double multiplier =
                        lower[depth] / diagonal[depth - 1];
                    multipliers[depth] = multiplier;
                    diagonal[depth] -= multiplier * upper[depth - 1];
                    right_hand_side[depth] -=
                        multiplier * right_hand_side[depth - 1];
                }
                symmetric_intensity[n_depth] =
                    right_hand_side[n_depth] / diagonal[n_depth];
                for (depth = n_depth; depth-- > 0;) {
                    symmetric_intensity[depth] =
                        (right_hand_side[depth]
                         - upper[depth] * symmetric_intensity[depth + 1])
                        / diagonal[depth];
                }

                for (index = 0; index < n_node * n_depth; ++index) {
                    tangent_rhs[index] =
                        -diagonal_response[index]
                        * symmetric_intensity[index / n_depth];
                }
                for (depth = 1; depth < n_node; ++depth) {
                    const Py_ssize_t row = depth * n_depth;
                    for (state = 0; state < n_depth; ++state) {
                        tangent_rhs[row + state] -=
                            lower_response[row + state]
                            * symmetric_intensity[depth - 1];
                    }
                }
                for (depth = 0; depth < n_depth; ++depth) {
                    const Py_ssize_t row = depth * n_depth;
                    for (state = 0; state < n_depth; ++state) {
                        tangent_rhs[row + state] -=
                            upper_response[row + state]
                            * symmetric_intensity[depth + 1];
                    }
                }
                for (depth = 0; depth < n_depth - 1; ++depth) {
                    tangent_rhs[(depth + 1) * n_depth + depth] +=
                        local_source_response[depth];
                }
                tangent_rhs[n_depth * n_depth + n_depth - 1] +=
                    local_source_response[n_depth - 1];
                for (depth = 1; depth < n_node; ++depth) {
                    const Py_ssize_t row = depth * n_depth;
                    const Py_ssize_t previous_row = (depth - 1) * n_depth;
                    for (state = 0; state < n_depth; ++state) {
                        tangent_rhs[row + state] -=
                            multipliers[depth]
                            * tangent_rhs[previous_row + state];
                    }
                }
                for (state = 0; state < n_depth; ++state) {
                    symmetric_response[n_depth * n_depth + state] =
                        tangent_rhs[n_depth * n_depth + state]
                        / diagonal[n_depth];
                }
                for (depth = n_depth; depth-- > 0;) {
                    const Py_ssize_t row = depth * n_depth;
                    const Py_ssize_t following_row = (depth + 1) * n_depth;
                    for (state = 0; state < n_depth; ++state) {
                        symmetric_response[row + state] =
                            (tangent_rhs[row + state]
                             - upper[depth]
                               * symmetric_response[following_row + state])
                            / diagonal[depth];
                    }
                }

                for (depth = 0; depth < n_depth; ++depth) {
                    const double optical_step =
                        expanded_tau[depth + 1] - expanded_tau[depth];
                    const double intensity_step =
                        symmetric_intensity[depth + 1]
                        - symmetric_intensity[depth];
                    const Py_ssize_t row = depth * n_depth;
                    const Py_ssize_t following_row = (depth + 1) * n_depth;
                    for (state = 0; state < n_depth; ++state) {
                        const double optical_step_response =
                            expanded_tau_response[following_row + state]
                            - expanded_tau_response[row + state];
                        const double intensity_step_response =
                            symmetric_response[following_row + state]
                            - symmetric_response[row + state];
                        integrated[row + state] += wavelength_weight[wave]
                            * ray_scale
                            * (intensity_step_response / optical_step
                               - intensity_step * optical_step_response
                                 / (optical_step * optical_step));
                    }
                }
            }
        }
        Py_END_ALLOW_THREADS
    }

    result = PyList_New(n_depth);
    if (result == NULL) {
        goto cleanup_feautrier_response;
    }
    for (depth = 0; depth < n_depth; ++depth) {
        PyObject *row = PyList_New(n_depth);
        if (row == NULL) {
            Py_CLEAR(result);
            goto cleanup_feautrier_response;
        }
        for (state = 0; state < n_depth; ++state) {
            PyObject *value = PyFloat_FromDouble(
                integrated[depth * n_depth + state]
            );
            if (value == NULL) {
                Py_DECREF(row);
                Py_CLEAR(result);
                goto cleanup_feautrier_response;
            }
            PyList_SET_ITEM(row, state, value);
        }
        PyList_SET_ITEM(result, depth, row);
    }

cleanup_feautrier_response:
    PyMem_Free(integrated);
    PyMem_Free(expanded_tau);
    PyMem_Free(expanded_source);
    PyMem_Free(expanded_tau_response);
    PyMem_Free(lower);
    PyMem_Free(diagonal);
    PyMem_Free(upper);
    PyMem_Free(lower_response);
    PyMem_Free(diagonal_response);
    PyMem_Free(upper_response);
    PyMem_Free(right_hand_side);
    PyMem_Free(multipliers);
    PyMem_Free(symmetric_intensity);
    PyMem_Free(tangent_rhs);
    PyMem_Free(symmetric_response);
    for (index = 0; index < 8; ++index) {
        if (views[index].obj != NULL) {
            PyBuffer_Release(&views[index]);
        }
    }
    return result;
}

static PyMethodDef module_methods[] = {
    {"emergent_flux", emergent_flux, METH_VARARGS,
     PyDoc_STR("emergent_flux(tau, source, mu, weight) -> list")},
    {"piecewise_linear_lorentz_convolution",
     piecewise_linear_lorentz_convolution,
     METH_VARARGS,
     PyDoc_STR("piecewise_linear_lorentz_convolution(native, profile, offset, gamma) -> list")},
    {"accumulate_lte_metal_line_profiles",
     accumulate_lte_metal_line_profiles,
     METH_VARARGS,
     PyDoc_STR("accumulate_lte_metal_line_profiles(wavelength, center, strength, sigma, gamma, half_window, population, absorption) -> None")},
    {"accumulate_metal_line_profiles",
     accumulate_metal_line_profiles,
     METH_VARARGS,
     PyDoc_STR("accumulate_metal_line_profiles(..., absorption, emissivity, retain_inverted) -> None")},
    {"metal_line_mean_intensity",
     metal_line_mean_intensity,
     METH_VARARGS,
     PyDoc_STR("metal_line_mean_intensity(wavelength, intensity, lambda_diagonal, center, sigma, gamma, mean_j, mean_lambda, include_lambda) -> None")},
    {"photoionization_rates",
     photoionization_rates,
     METH_VARARGS,
     PyDoc_STR("photoionization_rates(wavelength, intensity, cross_section, rate) -> None")},
    {"integrated_feautrier_interface_state_response",
     integrated_feautrier_interface_state_response,
     METH_VARARGS,
     PyDoc_STR("integrated_feautrier_interface_state_response(tau, source, source_response, opacity_response, mass, mu, weight, wavelength_weight) -> list")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_rt",
    "Compiled formal radiative-transfer kernels.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__rt(void)
{
    return PyModule_Create(&module_definition);
}
