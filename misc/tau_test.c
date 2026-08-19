#include <mpi.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum operation {
    OP_BARRIER,
    OP_BCAST,
    OP_GATHER,
    OP_SCATTER,
    OP_ALLGATHER,
    OP_ALLTOALL,
    OP_REDUCE,
    OP_ALLREDUCE,
    OP_SENDRECV
};

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s OP [--iterations N] [--count N]\n"
            "Operations: --barrier --bcast --gather --scatter\n"
            "            --allgather --alltoall --reduce --allreduce\n"
            "            --sendrecv\n",
            program);
}

static int positive_integer(const char *value, int *result)
{
    char *end;
    long parsed = strtol(value, &end, 10);

    if (*value == '\0' || *end != '\0' || parsed < 1 || parsed > INT_MAX) {
        return 0;
    }

    *result = (int) parsed;
    return 1;
}

static int parse_operation(const char *name, enum operation *operation)
{
    const char *names[] = {
        "--barrier", "--bcast", "--gather", "--scatter",
        "--allgather", "--alltoall", "--reduce", "--allreduce",
        "--sendrecv"
    };

    for (int i = 0; i < 9; ++i) {
        if (strcmp(name, names[i]) == 0) {
            *operation = (enum operation) i;
            return 1;
        }
    }

    return 0;
}

static const char *operation_name(enum operation operation)
{
    static const char *names[] = {
        "barrier", "bcast", "gather", "scatter", "allgather",
        "alltoall", "reduce", "allreduce", "sendrecv"
    };

    return names[operation];
}

int main(int argc, char **argv)
{
    int rank, size;
    int iterations = 1000;
    int count = 1024;
    enum operation operation;
    int valid = argc >= 2 && parse_operation(argv[1], &operation);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 2; valid && i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 || strcmp(argv[i], "--count") == 0) {
            int *target = strcmp(argv[i], "--iterations") == 0 ? &iterations : &count;
            if (i + 1 >= argc || !positive_integer(argv[++i], target)) {
                valid = 0;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            valid = 0;
        } else {
            valid = 0;
        }
    }

    if (!valid) {
        if (rank == 0) {
            usage(argv[0]);
        }
        MPI_Finalize();
        return 2;
    }

    size_t local_values = (size_t) count;
    size_t distributed_values = (size_t) size * local_values;
    int *sendbuf = NULL;
    int *recvbuf = NULL;

    switch (operation) {
    case OP_GATHER:
    case OP_SCATTER:
    case OP_ALLTOALL:
        sendbuf = malloc((operation == OP_GATHER ? local_values : distributed_values) * sizeof(*sendbuf));
        recvbuf = malloc((operation == OP_SCATTER ? local_values : distributed_values) * sizeof(*recvbuf));
        break;
    case OP_ALLGATHER:
        sendbuf = malloc(local_values * sizeof(*sendbuf));
        recvbuf = malloc(distributed_values * sizeof(*recvbuf));
        break;
    case OP_BCAST:
    case OP_ALLREDUCE:
    case OP_REDUCE:
    case OP_SENDRECV:
        sendbuf = malloc(local_values * sizeof(*sendbuf));
        recvbuf = malloc(local_values * sizeof(*recvbuf));
        break;
    case OP_BARRIER:
        break;
    }

    if ((operation != OP_BARRIER && (sendbuf == NULL || recvbuf == NULL))) {
        fprintf(stderr, "Rank %d could not allocate MPI buffers\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (size_t i = 0; i < local_values; ++i) {
        if (sendbuf != NULL) {
            sendbuf[i] = rank + (int) i;
        }
        if (recvbuf != NULL) {
            recvbuf[i] = 0;
        }
    }

    if (operation == OP_GATHER || operation == OP_SCATTER || operation == OP_ALLTOALL) {
        for (size_t i = 0; i < distributed_values; ++i) {
            if (sendbuf != NULL && operation != OP_GATHER) {
                sendbuf[i] = rank + (int) i;
            }
            if (recvbuf != NULL) {
                recvbuf[i] = 0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        switch (operation) {
        case OP_BARRIER:
            MPI_Barrier(MPI_COMM_WORLD);
            break;
        case OP_BCAST:
            MPI_Bcast(sendbuf, count, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        case OP_GATHER:
            MPI_Gather(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        case OP_SCATTER:
            MPI_Scatter(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        case OP_ALLGATHER:
            MPI_Allgather(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, MPI_COMM_WORLD);
            break;
        case OP_ALLTOALL:
            MPI_Alltoall(sendbuf, count, MPI_INT, recvbuf, count, MPI_INT, MPI_COMM_WORLD);
            break;
        case OP_REDUCE:
            MPI_Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            break;
        case OP_ALLREDUCE:
            MPI_Allreduce(sendbuf, recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            break;
        case OP_SENDRECV: {
            int destination = (rank + 1) % size;
            int source = (rank + size - 1) % size;
            MPI_Sendrecv(sendbuf, count, MPI_INT, destination, 0,
                         recvbuf, count, MPI_INT, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    double minimum, maximum, total;
    MPI_Reduce(&elapsed, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("operation=%s ranks=%d iterations=%d count=%d min=%g max=%g average=%g seconds\n",
               operation_name(operation), size, iterations, count,
               minimum, maximum, total / size);
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
