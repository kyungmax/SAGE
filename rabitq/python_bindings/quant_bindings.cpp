#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "bindings_common.hpp"
#include "rabitqlib/defines.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

static float ip_uint8_float(const float* query, const uint8_t* code, size_t dim) {
    float s = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        s += query[i] * static_cast<float>(code[i]);
    }
    return s;
}

py::array_t<float> rabitq_pairwise_est_dist(
    py::handle data,
    size_t nbits,
    const std::string& metric = "l2",
    bool fast_quantization = true
) {
    auto data_array = ensure_2d_array<float>(data, "data");

    const size_t n = static_cast<size_t>(data_array.shape(0));
    const size_t dim = static_cast<size_t>(data_array.shape(1));

    if (nbits < 1 || nbits > 8) {
        throw std::invalid_argument("nbits must be between 1 and 8");
    }

    rabitqlib::MetricType metric_type = metric_from_string(metric);

    // Same style as HNSW: use FhtKacRotator and pad dim to multiple of 64
    rabitqlib::Rotator<float>* rotator =
        rabitqlib::choose_rotator<float>(
            dim,
            rabitqlib::RotatorType::FhtKacRotator,
            rabitqlib::round_up_to_multiple(dim, 64)
        );

    const size_t padded_dim = rotator->size();

    std::vector<float> centroid(padded_dim, 0.0f);
    std::vector<float> rotated_data(n * padded_dim, 0.0f);

    // Rotate all data
    for (size_t i = 0; i < n; ++i) {
        rotator->rotate(
            data_array.data() + i * dim,
            rotated_data.data() + i * padded_dim
        );
    }

    // Single centroid = mean of rotated vectors
    for (size_t i = 0; i < n; ++i) {
        const float* x = rotated_data.data() + i * padded_dim;
        for (size_t d = 0; d < padded_dim; ++d) {
            centroid[d] += x[d];
        }
    }
    for (size_t d = 0; d < padded_dim; ++d) {
        centroid[d] /= static_cast<float>(n);
    }

    rabitqlib::quant::RabitqConfig config;
    if (fast_quantization) {
        config = rabitqlib::quant::faster_config(padded_dim, nbits);
    }

    std::vector<uint8_t> codes(n * padded_dim, 0);
    std::vector<float> f_add(n, 0.0f);
    std::vector<float> f_rescale(n, 0.0f);
    std::vector<float> f_error(n, 0.0f);

    // Quantize all data vectors
    for (size_t i = 0; i < n; ++i) {
        rabitqlib::quant::quantize_full_single<float, uint8_t>(
            rotated_data.data() + i * padded_dim,
            centroid.data(),
            padded_dim,
            nbits,
            codes.data() + i * padded_dim,
            f_add[i],
            f_rescale[i],
            f_error[i],
            metric_type,
            config
        );
    }

    const auto shape = std::vector<ssize_t>{
        static_cast<ssize_t>(n),
        static_cast<ssize_t>(n)
    };

    auto out = py::array_t<float>(shape);
    auto out_buf = out.mutable_unchecked<2>();

    // Pairwise estimated distance:
    // query = rotated raw vector
    // database = quantized code + factors
    for (size_t qi = 0; qi < n; ++qi) {
        const float* q = rotated_data.data() + qi * padded_dim;

        float q_to_centroid = 0.0f;
        float q_dot_centroid = 0.0f;

        if (metric_type == rabitqlib::METRIC_L2) {
            q_to_centroid = rabitqlib::euclidean_sqr(q, centroid.data(), padded_dim);
        } else {
            q_dot_centroid = rabitqlib::dot_product(q, centroid.data(), padded_dim);
        }

        float sum_q = 0.0f;
        for (size_t d = 0; d < padded_dim; ++d) {
            sum_q += q[d];
        }

        for (size_t xi = 0; xi < n; ++xi) {
            float g_add;
            float k1xsumq;

            if (metric_type == rabitqlib::METRIC_L2) {
                g_add = q_to_centroid;
                k1xsumq = -2.0f * sum_q;
            } else {
                g_add = q_dot_centroid;
                k1xsumq = sum_q;
            }

            float est = rabitqlib::quant::full_est_dist<float, uint8_t>(
                codes.data() + xi * padded_dim,
                q,
                ip_uint8_float,
                padded_dim,
                nbits,
                f_add[xi],
                f_rescale[xi],
                g_add,
                k1xsumq
            );

            out_buf(static_cast<ssize_t>(qi), static_cast<ssize_t>(xi)) = est;
        }
    }

    delete rotator;
    return out;
}

}  // namespace rabitqlib::python_bindings


void register_quant(py::module_ &m) {
    using namespace rabitqlib::python_bindings;

    m.def(
        "rabitq_pairwise_est_dist",
        &rabitq_pairwise_est_dist,
        py::arg("data"),
        py::arg("nbits"),
        py::arg("metric") = "l2",
        py::arg("fast_quantization") = true,
        "Compute pairwise RaBitQ estimated distances without HNSW."
    );
}
