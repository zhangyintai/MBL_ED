#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iterator>
#include <map>
#include <memory.h>
#include "mkl.h"
#include <mathimf.h>
#include <mpi.h>

using namespace std;

#define Max(a, b) ((a) < (b) ? (b) : (a))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Abs(a) ((a) > (0) ? (a) : -(a))

int StringToInt(char *a);
long int StringToLongInt(char *a);
double StringToDouble(char *a);

int BinToDec(int *a, int L);
int DecToBin(int *a, int L, long int s);
void BuildStates(int L, int occupied, map<long int, long int> *states, vector<long int> *states_,
                 long int *num_of_states, int num_of_1, int *temp, int temp_len);
void CalculateE0(double *E0, map<long int, long int> states,
                 int L, double V, double W, char random_mode, double para);
void BuildHamiltonian(map<long int, long int> states,
                      long int num_of_states,
                      int L, double *H, double *E0, double t);
void PrintHamiltonian(double *H, long int num_of_states);

void EigenSolver(double *H, double *eigen_e, long int num_of_states);
double EntanglementEntropy(double *eigenstates, long int num_of_states, vector<long int> states_, int L);

long int FindCenter(double *a, long int n);
double RadiusNN(double *wavf, map<long int, long int> states,
                vector<long int> states_, long int num_of_states, int L);
void AverageResult(int L, int N_random, double W, double t, double V,
                   char random_mode, double para, double *result, char ent_entro_cal);
double GapRatio(double *E, long int num_of_states);
void Eigen(int L, long int N, double *a, double *w, char mode, int print);
void DGESVD(long int N, double *a, double *w, int print);

extern void print_matrix(char *desc, MKL_INT64 m, MKL_INT64 n, double *a, MKL_INT64 lda);

void ExpRandom(int L, double *random, double W, double alpha);
void MultiPointRandom(int L, double *random, double W, double n);

long int SEED_random = 20220316;
const long int m_random = 2147483647;          //set m=a*q+r;  randomz: 100000001; 16807: 2147483647
const long int a_random = 16807;               //set a; rand	omz: 329; 16807: 16807
const long int q_random = m_random / a_random; //set q
const long int r_random = m_random % a_random; //set r
const double pi = 3.141592653589793;
const double e = 2.718281828459045;
const long int b_random = 0;
long int Schrage(long int z);
double Random();

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int num_of_procs, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs);

    SEED_random = SEED_random + rank;

    int L, N_random, N_w;
    double t, V, W_min, W_max;
    char random_mode, ent_entro_cal;
    double para;

    L = StringToInt(argv[1]);
    t = StringToDouble(argv[2]);
    V = StringToDouble(argv[3]);
    W_min = StringToDouble(argv[4]);
    W_max = StringToDouble(argv[5]);
    N_w = StringToDouble(argv[6]);
    N_random = StringToInt(argv[7]);
    random_mode = *(argv[8]);
    para = StringToDouble(argv[9]);
    ent_entro_cal = *(argv[10]);

    int N_group = num_of_procs / N_w;

    if (N_w == 1)
    {
        double *result = new double[2];
        printf("L: %d, t: %lf, V: %lf, W: %lf, N_w: %d, N_random: %d, Random_mode: %c, para: %lf\n\n",
               L, t, V, W_max, N_w, N_random, random_mode, para);
        AverageResult(L, N_random, W_max, t, V, random_mode, para, result, ent_entro_cal);
        printf("%lf\n\n%lf\n\n", result[0], result[1]);

        delete result;

        MPI_Finalize();
        exit(0);

        return 0;
    }

    double W = (W_max - W_min) / ((double)(N_w - 1)) * ((double)(rank % N_w)) + W_min;

    double *result = new double[2];
    double *RESULT = new double[num_of_procs * 2];
    int *disp = new int[num_of_procs * 2], *recv_count = new int[num_of_procs * 2];

    for (int i = 0; i < num_of_procs; i++)
    {
        disp[i] = i * 2;
        recv_count[i] = 2;
    }

    memset(RESULT, 0, sizeof(double) * num_of_procs * 2);
    AverageResult(L, N_random, W, t, V, random_mode, para, result, ent_entro_cal);

    MPI_Gatherv(result, 2, MPI_DOUBLE, RESULT, recv_count, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("L: %d, t: %lf, V: %lf, W_min: %lf, W_max: %lf, N_w: %d, N_random: %d, N_group: %d, Random_mode: %c, para: %lf\n\n",
               L, t, V, W_min, W_max, N_w, N_random, N_group, random_mode, para);
        printf("[");

        for (int i = 0; i < num_of_procs; i++)
        {
            printf("%.9lf, ", RESULT[i * 2]);
        }
        printf("]\n\n");

        printf("[");

        for (int i = 0; i < num_of_procs; i++)
        {
            printf("%.9lf, ", RESULT[i * 2 + 1]);
        }
        printf("]\n\n");
    }

    delete result, RESULT, disp, recv_count;

    MPI_Finalize();
    exit(0);

    return 0;
}

int StringToInt(char *a)
{
    int s = 0, len = strlen(a);
    for (int i = 0; i < strlen(a); i++)
    {
        if (int(a[i]) >= 48 && int(a[i]) <= 57)
        {
            s = s * 10;
            s = s + a[i] - 48;
        }
    }

    return s;
}

long int StringToLongInt(char *a)
{
    long int s = 0;
    int len = strlen(a);

    for (int i = 0; i < strlen(a); i++)
    {
        if (int(a[i]) >= 48 && int(a[i]) <= 57)
        {
            s = s * 10;
            s = s + a[i] - 48;
        }
    }

    return s;
}

double StringToDouble(char *a)
{
    double s = 0;
    int len = strlen(a);

    double decimal = 1;

    for (int i = 0; i < strlen(a); i++)
    {
        if (int(a[i]) >= 48 && int(a[i]) <= 57)
        {
            if (decimal == 1)
                s = s * 10.0;

            s = s + (a[i] - 48) * decimal;

            if (decimal != 1)
            {
                decimal = decimal / 10;
            }
        }
        if (a[i] == '.')
        {
            decimal = decimal / 10.0;
        }
    }

    return s;
}

int BinToDec(int *a, int L)
{
    int ii;

    long int s = 0;

    for (ii = 0; ii < L; ii++)
    {
        if (*(a + ii) == 1)
            s = s + (1 << ii);
    }

    return s;
}

int DecToBin(int *a, int L, long int s)
{
    int ii;
    int num_of_1 = 0;

    memset(a, 0, sizeof(int) * L);

    for (ii = 0; ii < L; ii++)
    {
        if (s == 0)
            break;

        a[ii] = s % 2;
        if (a[ii] == 1)
        {
            num_of_1++;
        }

        s = s >> 1;
    }

    return num_of_1;
}

void BuildStates(int L, int occupied, map<long int, long int> *states, vector<long int> *states_,
                 long int *num_of_states, int num_of_1, int *temp, int temp_len)
{

    if (temp_len == L)
    {
        return;
    }

    if (num_of_1 == occupied)
    {
        for (int ii = temp_len; ii < L; ii++)
        {
            temp[ii] = 0;
        }

        long int temp_ = BinToDec(temp, L);
        (*states)[temp_] = *num_of_states;
        states_->push_back(temp_);

        *num_of_states = (*num_of_states) + 1;

        return;
    }
    else if ((temp_len - num_of_1) == (L - occupied))
    {
        for (int ii = temp_len; ii < L; ii++)
        {
            temp[ii] = 1;
        }

        long int temp_ = BinToDec(temp, L);
        (*states)[temp_] = *num_of_states;
        states_->push_back(temp_);

        *num_of_states = (*num_of_states) + 1;

        return;
    }
    else
    {
        temp[temp_len] = 1;
        BuildStates(L, occupied, states, states_, num_of_states, num_of_1 + 1, temp, temp_len + 1);

        temp[temp_len] = 0;
        BuildStates(L, occupied, states, states_, num_of_states, num_of_1, temp, temp_len + 1);
    }
}

void CalculateE0(double *E0, map<long int, long int> states,
                 int L, double V, double W, char random_mode, double para)
{
    double *random = new double[L];
    memset(random, 0, sizeof(double) * L);

    if (random_mode == 'M')
    {
        MultiPointRandom(L, random, W, para);
    }
    else if (random_mode == 'E')
    {
        ExpRandom(L, random, W, para);
    }
    else
    {
        for (int i = 0; i < L; i++)
        {
            random[i] = W * Random() - W / 2;
        }
    }

    // for (int i = 0; i < L; i++)
    // {
    //     printf("%lf ", random[i]);
    // }
    // printf("\n");

    for (map<long int, long int>::iterator ii = states.begin(); ii != states.end(); ii++)
    {
        int *temp = new int[L];
        int a = DecToBin(temp, L, ii->first);

        E0[ii->second] = 0;

        for (int j = 0; j < L; j++)
        {
            if (temp[j] == 1)
            {
                E0[ii->second] = E0[ii->second] + random[j];
            }

            if (j != 0)
            {
                if ((temp[j] == 1) && (temp[j - 1] == 1))
                {
                    E0[ii->second] = E0[ii->second] + V;
                }
            }
            else
            {
                if ((temp[0]  == 1) && (temp[L - 1] == 1))
                {
		    E0[ii->second] = E0[ii->second] + V;
		}
            }
        }

        delete temp;
    }

    delete random;
}

void BuildHamiltonian(map<long int, long int> states,
                      long int num_of_states,
                      int L, double *H, double *E0, double t)
{
    for (long int i = 0; i < num_of_states; i++)
    {
        H[num_of_states * i + i] = E0[i];
    }

    for (map<long int, long int>::iterator ii = states.begin(); ii != states.end(); ii++)
    {
        // 1-D chain NN hopping
        for (int jj = 0; jj < L - 1; jj++)
        {
            map<long int, long int>::iterator temp = states.find((1 << jj) + (ii->first));
            if (temp == states.end())
                continue;

            H[num_of_states * temp->second + ii->second] = t;
        }
        //PBC
        map<long int, long int>::iterator temp = states.find((1 << (L-1)) - 1 + (ii->first));
        if (temp != states.end())
            H[num_of_states * temp->second + ii->second] = t;
    }
}

void PrintHamiltonian(double *H, long int num_of_states)
{
    for (long int ii = 0; ii < num_of_states; ii++)
    {
        for (long int jj = 0; jj < num_of_states; jj++)
        {
            printf("%.6lf\t", *(H + ii * num_of_states + jj));
        }
        printf("\n");
    }
    printf("\n");
}

void EigenSolver(double *H, double *eigen_e, long int num_of_states)
{
    long int LDA = num_of_states;

    MKL_INT64 n = num_of_states, lda = LDA, info;

    /* Executable statements */
    /* Solve eigenproblem */
    if (num_of_states <= 10000)
    {
        info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, H, lda, eigen_e);
    }
    else
    {
        info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, H, lda, eigen_e);
    }
    /* Check for convergence */
    if (info > 0)
    {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }
}

long int FindCenter(double *a, long int n)
{
    double temp_max = (*a) * (*a);

    long int temp_max_i = 0;
    for (long int i = 1; i < n; i++)
    {
        // printf("(%lf, %lf, %d) ", (*(a + i)) * (*(a + i)), temp_max, temp_max_i);
        if ((*(a + i)) * (*(a + i)) > temp_max)
        {
            temp_max = (*(a + i)) * (*(a + i));
            temp_max_i = i;
        }
    }
    // printf("\n");

    return temp_max_i;
}

double RadiusNN(double *wavf, map<long int, long int> states,
                vector<long int> states_, long int num_of_states, int L)
{
    double radius = 0;

    long int center = FindCenter(wavf, num_of_states);
    int *a = new int[L];

    for (map<long int, long int>::iterator ii = states.begin(); ii != states.end(); ii++)
    {

        long int temp_2 = ii->first;
        long int temp_1 = states_.at(center);

        int steps = 0;

        while (temp_2 != temp_1)
        {
            steps = steps + Abs(((int)log2(temp_1)) - ((int)log2(temp_2)));

            temp_1 = temp_1 - (1 << (int)log2(temp_1));
            temp_2 = temp_2 - (1 << (int)log2(temp_2));
        }

        radius = radius + steps * (*(wavf + ii->second)) *
                              (*(wavf + ii->second));
        // printf("%d, %d, %d, %lf\n", temp_1, center, DecToBin(a, L, temp_2), (*(wavf + ii->second)) * (*(wavf + ii->second)));
    }

    // printf("%lf \n", radius);

    delete a;

    return radius;
}

void AverageResult(int L, int N_random, double W, double t, double V,
                   char random_mode, double para, double *result, char ent_entro_cal)
{
    double *H, *E0;
    long int num_of_states = 0;
    map<long int, long int> *states = new map<long int, long int>;
    vector<long int> *states_ = new vector<long int>;
    int *temp = new int[L];

    double radius_aver = 0;

    BuildStates(L, L / 2, states, states_, &num_of_states, 0, temp, 0);

    // for (map<long int, long int>::iterator ii = states->begin(); ii != states->end(); ii++)
    // {
    //     printf("%d %d\n", ii->first, ii->second);
    // }
    // printf("_________________________________________________\n\n");

    H = new double[num_of_states * num_of_states];
    E0 = new double[num_of_states];

    result[0] = 0;
    result[1] = 0;

    for (int ii = 0; ii < N_random; ii++)
    {
        memset(H, 0, sizeof(double) * num_of_states * num_of_states);
        memset(E0, 0, sizeof(double) * num_of_states);

        CalculateE0(E0, *states, L, V, W, random_mode, para);
        BuildHamiltonian(*states, num_of_states, L, H, E0, t);

        // PrintHamiltonian(H, num_of_states);

        EigenSolver(H, E0, num_of_states);

        // for (long int jj = num_of_states / 3; jj < num_of_states / 3 * 2; jj++)
        // {
        //    radius_aver = radius_aver + RadiusNN(H + jj * num_of_states, *states, *states_, num_of_states, L);
        // }

        result[0] = result[0] + GapRatio(E0, num_of_states);
        if (ent_entro_cal == 'Y')
        {
            result[1] = result[1] + EntanglementEntropy(H, num_of_states, *states_, L);
        }
    }

    result[0] = result[0] / N_random;
    result[1] = result[1] / N_random;

    delete H;
    delete E0;
    delete temp;

    states->clear();
    delete states;

    states_->clear();
    delete states_;
}

double GapRatio(double *E, long int num_of_states)
{
    double gap_ratio = 0, gap_1, gap_2;
    for (long int i = num_of_states / 3; i <= num_of_states / 3 * 2; i++)
    {
        gap_1 = Abs(E[i] - E[i - 1]);
        gap_2 = Abs(E[i + 1] - E[i]);
        gap_ratio = gap_ratio + Min(gap_1, gap_2) / Max(gap_1, gap_2);
    }

    return gap_ratio * 3.0 / num_of_states;
}

double EntanglementEntropy(double *eigenstates, long int num_of_states, vector<long int> states_, int L)
{
    long int size_phi = (long int)pow(2, L), size_w = (long int)pow(2, L / 2);

    double *phi = new double[size_phi], *w = new double[size_w];

    double epsilon = 1e-9;

    double s = 0;
    for (long int i = num_of_states / 3; i <= num_of_states / 3 * 2; i++)
    {
        memset(phi, 0, sizeof(double) * size_phi);
        memset(w, 0, sizeof(double) * size_w);

        for (long int j = 0; j < num_of_states; j++)
        {
            *(phi + states_.at(j)) = *(eigenstates + i * num_of_states + j);
        }

        DGESVD(size_w, phi, w, 0);

        for (long int j = 0; j < size_w; j++)
        {
            if (*(w + j) >= epsilon)
            {
                s = s - 2.0 * (*(w + j)) * (*(w + j)) * log2(*(w + j));
            }
        }
    }

    delete phi;
    delete w;

    return s / (double)num_of_states * 3;
}

void DGESVD(long int N, double *a, double *w, int print)
{
    /* Locals */
    MKL_INT m = N, n = N, lda = N, ldu = N, ldvt = N, info, lwork;
    double wkopt;
    double *work;
    /* Local arrays */
    double *u = (double *)malloc(sizeof(double) * N * N);
    double *vt = (double *)malloc(sizeof(double) * N * N);
    memset(u, 0, sizeof(u) * N * N);
    memset(vt, 0, sizeof(double) * N * N);
    // double a[LDA * N] = {
    //     8.79, 6.11, -9.15, 9.57, -3.49, 9.84,
    //     9.93, 6.91, -7.93, 1.64, 4.02, 0.15,
    //     9.83, 5.04, 4.86, 8.83, 9.80, -8.99,
    //     5.45, -0.27, 4.85, 0.74, 10.00, -6.02,
    //     3.16, 7.98, 3.01, 5.80, 4.27, -5.31};
    /* Executable statements */
    // printf(" DGESVD Example Program Results\n");
    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgesvd("N", "N", &m, &n, a, &lda, w, u, &ldu, vt, &ldvt, &wkopt, &lwork,
           &info);
    lwork = (MKL_INT)wkopt;
    work = (double *)malloc(lwork * sizeof(double));
    /* Compute SVD */
    dgesvd("N", "N", &m, &n, a, &lda, w, u, &ldu, vt, &ldvt, work, &lwork,
           &info);
    /* Check for convergence */
    if (info > 0)
    {
        printf("The algorithm computing SVD failed to converge.\n");
        // exit(1);
    }
    // /* Print singular values */
    // print_matrix("Singular values", 1, n, s, 1);
    // /* Print left singular vectors */
    // print_matrix("Left singular vectors (stored columnwise)", m, n, u, ldu);
    // /* Print right singular vectors */
    // print_matrix("Right singular vectors (stored rowwise)", n, n, vt, ldvt);
    /* Free workspace */
    free((void *)work);
    free(vt);
    free(u);
    // exit(0);
} /* End of DGESVD Example */

double Random()
{
    int temp;

    temp = Schrage(SEED_random) + b_random;

    if (temp > m_random)
        temp = temp - m_random;

    SEED_random = temp;

    return (double)(temp) / 2147483647.0;
}

long int Schrage(long int z)
{
    int s;

    s = a_random * (z % q_random) - r_random * (z / q_random);
    if (s > 0)
        return s;
    else
        return s + m_random;
}

void ExpRandom(int L, double *random, double W, double alpha)
{
    int i;
    double temp = Random();

    for (i = 0; i < L; i++)
    {
        temp = Random();
        if (alpha == 0)
        {
            *(random + i) = (temp * W - W / 2.0) * 2.0;
        }

        else
        {
            if (temp == 0.5)
            {
                *(random + i) = 0;
            }
            else if (temp < 0.5)
            {
                temp = pow(e, alpha) - 2.0 * temp * (pow(e, alpha) - 1.0);
                *(random + i) = -W / alpha * log(temp);
            }
            else
            {
                temp = temp - 0.5;
                temp = 2.0 * temp * (pow(e, alpha) - 1.0) + 1;
                *(random + i) = W / alpha * log(temp);
            }
        }
    }
}

void MultiPointRandom(int L, double *random, double W, double n)
{
    int i, j;
    double temp = Random();

    for (i = 0; i < L; i++)
    {
        temp = Random();
        for (j = 0; j < (int)n; j++)
        {
            if (temp < ((double)j + 1.0) / (double)n)
            {
                *(random + i) = W * 2.0 * (double)j / ((double)n - 1.0) - W;
                break;
            }
        }
    }
}
