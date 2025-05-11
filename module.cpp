#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include <omp.h>

// ------------------------------------  //
//     warmup                            //
// ------------------------------------ //

inline float twoDimRead(const std::vector<float> &T, int i, int j, int cols)
{
    return T[i * cols + j];
}
inline void twoDimWrite(std::vector<float> &T, int i, int j, int cols, float v)
{
    T[i * cols + j] = v;
}

inline float fourDimRead(const std::vector<float> &T,
                         int b, int h, int n, int d,
                         int B, int H, int N, int D)
{
    return T[((b * H + h) * N + n) * D + d];
}
inline void fourDimWrite(std::vector<float> &T,
                         int b, int h, int n, int d,
                         int B, int H, int N, int D, float v)
{
    T[((b * H + h) * N + n) * D + d] = v;
}

static std::vector<float> formatTensor(torch::Tensor t)
{
    t = t.flatten().contiguous();
    return {t.data_ptr<float>(), t.data_ptr<float>() + t.numel()};
}

// -------------------------------------------------------- //
//   PART 1                                                 //
// -------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor Q_, torch::Tensor K_, torch::Tensor V_, torch::Tensor QKt_,
                               int B, int H, int N, int D)
{
    auto O_ = at::zeros({B, H, N, D}, at::kFloat);
    auto Q = formatTensor(Q_), K = formatTensor(K_), V = formatTensor(V_), QKt = formatTensor(QKt_), O = formatTensor(O_);

    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    twoDimWrite(QKt, i, j, N, 0.0f);

            for (int i = 0; i < N; i++)
                for (int k = 0; k < D; k++)
                {
                    float qv = fourDimRead(Q, b, h, i, k, B, H, N, D);
                    for (int j = 0; j < N; j++)
                        twoDimWrite(QKt, i, j, N,
                                    twoDimRead(QKt, i, j, N) + qv * fourDimRead(K, b, h, j, k, B, H, N, D));
                }
            for (int i = 0; i < N; i++)
            {
                float s = 0;
                for (int j = 0; j < N; j++)
                {
                    float e = std::exp(twoDimRead(QKt, i, j, N));
                    twoDimWrite(QKt, i, j, N, e);
                    s += e;
                }
                for (int j = 0; j < N; j++)
                {
                    twoDimWrite(QKt, i, j, N, twoDimRead(QKt, i, j, N) / s);
                }
            }
            for (int i = 0; i < N; i++)
                for (int d = 0; d < D; d++)
                {
                    float acc = 0;
                    for (int j = 0; j < N; j++)
                        acc += twoDimRead(QKt, i, j, N) * fourDimRead(V, b, h, j, d, B, H, N, D);
                    fourDimWrite(O, b, h, i, d, B, H, N, D, acc);
                }
        }
    return torch::from_blob(O.data(), {B, H, N, D}, torch::kFloat32).clone();
}

// ---------------------------------------------------------------- //
//   PART 2                                                         //
// ---------------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor Q_, torch::Tensor K_, torch::Tensor V_, torch::Tensor QKt_,
                                        int B, int H, int N, int D)
{
    auto O_ = at::zeros({B, H, N, D}, at::kFloat);
    auto Q = formatTensor(Q_), K = formatTensor(K_), V = formatTensor(V_), QKt = formatTensor(QKt_), O = formatTensor(O_);
    const int BLOCK = 64;
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    twoDimWrite(QKt, i, j, N, 0.0f);

            for (int ii = 0; ii < N; ii += BLOCK)
            {
                int iend = std::min(ii + BLOCK, N);
                for (int kk = 0; kk < D; kk += BLOCK)
                {
                    int kend = std::min(kk + BLOCK, D);
                    for (int jj = 0; jj < N; jj += BLOCK)
                    {
                        int jend = std::min(jj + BLOCK, N);
                        for (int i = ii; i < iend; i++)
                            for (int k = kk; k < kend; k++)
                            {
                                float qv = fourDimRead(Q, b, h, i, k, B, H, N, D);
                                for (int j = jj; j < jend; j++)
                                {
                                    float acc = twoDimRead(QKt, i, j, N) + qv * fourDimRead(K, b, h, j, k, B, H, N, D);
                                    twoDimWrite(QKt, i, j, N, acc);
                                }
                            }
                    }
                }
            }
            for (int i = 0; i < N; i++)
            {
                float s = 0;
                for (int j = 0; j < N; j++)
                {
                    float e = std::exp(twoDimRead(QKt, i, j, N));
                    twoDimWrite(QKt, i, j, N, e);
                    s += e;
                }
                for (int j = 0; j < N; j++)
                    twoDimWrite(QKt, i, j, N, twoDimRead(QKt, i, j, N) / s);
            }
            for (int ii = 0; ii < N; ii += BLOCK)
            {
                int iend = std::min(ii + BLOCK, N);
                for (int jj = 0; jj < N; jj += BLOCK)
                {
                    int jend = std::min(jj + BLOCK, N);
                    for (int kk = 0; kk < D; kk += BLOCK)
                    {
                        int kend = std::min(kk + BLOCK, D);
                        for (int i = ii; i < iend; i++)
                            for (int j = jj; j < jend; j++)
                            {
                                float pij = twoDimRead(QKt, i, j, N);
                                for (int d = kk; d < kend; d++)
                                {
                                    float acc = fourDimRead(O, b, h, i, d, B, H, N, D) + pij * fourDimRead(V, b, h, j, d, B, H, N, D);
                                    fourDimWrite(O, b, h, i, d, B, H, N, D, acc);
                                }
                            }
                    }
                }
            }
        }
    return torch::from_blob(O.data(), {B, H, N, D}, torch::kFloat32).clone();
}

// -------------------------------------------------------- //
//   PART 3                                                 //
// -------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor Q_, torch::Tensor K_, torch::Tensor V_, torch::Tensor temp_,
                               int B, int H, int N, int D)
{
    auto O_ = at::zeros({B, H, N, D}, at::kFloat);
    auto Q = formatTensor(Q_), K = formatTensor(K_), V = formatTensor(V_), O = formatTensor(O_);

    auto temp = formatTensor(temp_);
    int T = omp_get_max_threads();
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            int tid = omp_get_thread_num();
            float *rowBuf = temp.data() + tid * N;
            for (int i = 0; i < N; i++)
            {
                float sum = 0;
                for (int j = 0; j < N; j++)
                {
                    float acc = 0;
                    for (int k = 0; k < D; k++)
                        acc += fourDimRead(Q, b, h, i, k, B, H, N, D) * fourDimRead(K, b, h, j, k, B, H, N, D);
                    rowBuf[j] = std::exp(acc);
                    sum += rowBuf[j];
                }
                for (int j = 0; j < N; j++)
                    rowBuf[j] /= sum;
                for (int d = 0; d < D; d++)
                {
                    float acc = 0;
                    for (int j = 0; j < N; j++)
                        acc += rowBuf[j] * fourDimRead(V, b, h, j, d, B, H, N, D);
                    fourDimWrite(O, b, h, i, d, B, H, N, D, acc);
                }
            }
        }
    return torch::from_blob(O.data(), {B, H, N, D}, torch::kFloat32).clone();
}

// -------------------------------------------------------- //
//   PART 4                                                 //
// -------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor Q_, torch::Tensor K_, torch::Tensor V_,
                               torch::Tensor Qi_, torch::Tensor Kj_, torch::Tensor Vj_,
                               torch::Tensor Sij_, torch::Tensor Pij_, torch::Tensor PV_,
                               torch::Tensor Oi_, torch::Tensor L_, torch::Tensor Li_,
                               torch::Tensor Lij_, torch::Tensor Lnew_,
                               int Bc, int Br, int B, int H, int N, int D)
{
    auto O_ = at::zeros({B, H, N, D}, at::kFloat);
    auto Q = formatTensor(Q_), K = formatTensor(K_), V = formatTensor(V_), O = formatTensor(O_);
    auto Qi = formatTensor(Qi_), Kj = formatTensor(Kj_), Vj = formatTensor(Vj_);
    auto Sij = formatTensor(Sij_), Pij = formatTensor(Pij_), PV = formatTensor(PV_);
    auto Oi = formatTensor(Oi_), l = formatTensor(L_), li = formatTensor(Li_);
    auto lij = formatTensor(Lij_), lnew = formatTensor(Lnew_);

    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < N; i++)
            {
                l[i] = -std::numeric_limits<float>::infinity();
            }
            for (int i0 = 0; i0 < N; i0 += Br)
            {
                int iLen = std::min(Br, N - i0);

                for (int i = 0; i < iLen; i++)
                {
                    for (int d = 0; d < D; d++)
                    {
                        Qi[i * D + d] = fourDimRead(Q, b, h, i0 + i, d, B, H, N, D);
                    }
                }

                for (int i = 0; i < iLen; i++)
                {
                    li[i] = l[i0 + i];

                    for (int d = 0; d < D; d++)
                    {
                        PV[i * D + d] = 0.0f;
                    }
                }

                for (int j0 = 0; j0 < N; j0 += Bc)
                {
                    int jLen = std::min(Bc, N - j0);

                    for (int j = 0; j < jLen; j++)
                    {
                        for (int d = 0; d < D; d++)
                        {
                            Kj[j * D + d] = fourDimRead(K, b, h, j0 + j, d, B, H, N, D);
                            Vj[j * D + d] = fourDimRead(V, b, h, j0 + j, d, B, H, N, D);
                        }
                    }

                    for (int i = 0; i < iLen; i++)
                    {
                        lij[i] = -std::numeric_limits<float>::infinity();

                        for (int j = 0; j < jLen; j++)
                        {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                            {
                                dot += Qi[i * D + d] * Kj[j * D + d];
                            }
                            Sij[i * jLen + j] = dot;
                            lij[i] = std::max(lij[i], dot);
                        }
                    }

                    for (int i = 0; i < iLen; i++)
                    {
                        float lnew_val = std::max(li[i], lij[i]);

                        float scale = (li[i] == -std::numeric_limits<float>::infinity()) ? 1.0f : std::exp(li[i] - lnew_val);

                        for (int d = 0; d < D; d++)
                        {
                            PV[i * D + d] *= scale;
                        }

                        float sum_exp = 0.0f;
                        for (int j = 0; j < jLen; j++)
                        {
                            float exp_val = std::exp(Sij[i * jLen + j] - lnew_val);
                            Pij[i * jLen + j] = exp_val;
                            sum_exp += exp_val;

                            for (int d = 0; d < D; d++)
                            {
                                PV[i * D + d] += exp_val * Vj[j * D + d];
                            }
                        }

                        li[i] = lnew_val;
                        l[i0 + i] = lnew_val;
                    }
                }

                for (int i = 0; i < iLen; i++)
                {
                    float m = li[i];
                    float denom = 0.0f;

                    for (int j0 = 0; j0 < N; j0 += Bc)
                    {
                        int jLen = std::min(Bc, N - j0);

                        for (int j = 0; j < jLen; j++)
                        {
                            for (int d = 0; d < D; d++)
                            {
                                Kj[j * D + d] = fourDimRead(K, b, h, j0 + j, d, B, H, N, D);
                            }
                        }

                        for (int j = 0; j < jLen; j++)
                        {
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++)
                            {
                                dot += Qi[i * D + d] * Kj[j * D + d];
                            }
                            denom += std::exp(dot - m);
                        }
                    }

                    for (int d = 0; d < D; d++)
                    {
                        Oi[i * D + d] = PV[i * D + d] / denom;
                    }
                }

                for (int i = 0; i < iLen; i++)
                {
                    for (int d = 0; d < D; d++)
                    {
                        fourDimWrite(O, b, h, i0 + i, d, B, H, N, D, Oi[i * D + d]);
                    }
                }
            }
        }

    return torch::from_blob(O.data(), {B, H, N, D}, torch::kFloat32).clone();
}

// ------------------------------------ //
//       PYBIND11: BIND EVERYTHING      //
// ------------------------------------ //

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("twoDimRead", &twoDimRead, "2D Read");
    m.def("fourDimRead", &fourDimRead, "4D Read");
    m.def("myNaiveAttention", &myNaiveAttention, "Part1: Naive");
    m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, "Part2: Blocked+Softmax");
    m.def("myFusedAttention", &myFusedAttention, "Part3: Fused+OpenMP");
    m.def("myFlashAttention", &myFlashAttention, "Part4: FlashAttention");
}
