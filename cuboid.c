/* This file is part of dolfinx-benchmarks.
 *
 * Copyright (C) 2022 James D. Trotter
 *
 * cuboid is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cuboid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cuboid.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-08-15
 *
 * Generate a cuboid mesh.
 */

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

#include <errno.h>

#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char * program_name = "cuboid";
const char * program_version = "1.0.0";
const char * program_copyright =
    "Copyright (C) 2022 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
#ifndef _GNU_SOURCE
const char * program_invocation_name;
const char * program_invocation_short_name;
#endif

enum meshformat
{
    meshformat_carp_pts = 0,
    meshformat_carp_elem,
    meshformat_carp_lon,
    meshformat_tetgen_node,
    meshformat_tetgen_ele,
    meshformat_dolfin_xdmf_ascii,
    meshformat_dolfin_xdmf_hdf5
};

/**
 * ‘program_options’ contains data to related program options.
 */
struct program_options
{
    double xmin, xmax;  /* extent of the cuboid along the x-axis */
    double ymin, ymax;  /* extent of the cuboid along the y-axis */
    double zmin, zmax;  /* extent of the cuboid along the z-axis */
    double dx, dy, dz;  /* spatial resolutions in the x-, y- and z-direction */
    enum meshformat meshformat;
    char * hdf5path;
    bool quiet;
    int verbose;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    /* default to a 20000x7000x3000 micrometer cuboid with a
     * resolution of 100 micrometres. */
    args->xmin =     0.0;
    args->xmax = 20000.0;
    args->ymin =     0.0;
    args->ymax =  7000.0;
    args->zmin =     0.0;
    args->zmax =  3000.0;
    args->dx = 100.0;
    args->dy = 100.0;
    args->dz = 100.0;
    args->meshformat = meshformat_dolfin_xdmf_ascii;
    args->hdf5path = NULL;
    args->quiet = false;
    args->verbose = 0;
    return 0;
}

/**
 * ‘program_options_free()’ frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
    if (args->hdf5path) free(args->hdf5path);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..]\n", program_name);
    fprintf(f, "\n");
    fprintf(f, " Generate a cuboid mesh with optional fibre directions.\n");
    fprintf(f, "\n");
    fprintf(f, " Options are:\n");
    fprintf(f, "  --xmin=DOUBLE\t\tminimum extent in the x-direction (default: 0)\n");
    fprintf(f, "  --xmax=DOUBLE\t\tmaximum extent in the x-direction (default: 20 000)\n");
    fprintf(f, "  --ymin=DOUBLE\t\tminimum extent in the y-direction (default: 0)\n");
    fprintf(f, "  --ymax=DOUBLE\t\tmaximum extent in the y-direction (default: 7 000)\n");
    fprintf(f, "  --zmin=DOUBLE\t\tminimum extent in the z-direction (default: 0)\n");
    fprintf(f, "  --zmax=DOUBLE\t\tmaximum extent in the z-direction (default: 3 000)\n");
    fprintf(f, "  --dx=DOUBLE\t\tspatial resolution in the x-direction (default: 100)\n");
    fprintf(f, "  --dy=DOUBLE\t\tspatial resolution in the y-direction (default: 100)\n");
    fprintf(f, "  --dz=DOUBLE\t\tspatial resolution in the z-direction (default: 100)\n");
    fprintf(f, "\n");
    fprintf(f, "  --carp-pts\t\t\toutput mesh nodes in CARP ASCII format\n");
    fprintf(f, "  --carp-elem\t\t\toutput mesh elements in CARP ASCII format\n");
    fprintf(f, "  --carp-lon\t\t\toutput fibre orientations in CARP ASCII format\n");
    fprintf(f, "  --tetgen-node\t\t\toutput mesh nodes in Tetgen ASCII format\n");
    fprintf(f, "  --tetgen-ele\t\t\toutput mesh elements in Tetgen ASCII format\n");
    fprintf(f, "  --dolfin-xdmf-ascii\t\toutput mesh in DOLFIN XDMF (ASCII) format\n");
    fprintf(f, "  --dolfin-xdmf-hdf5=FILE\toutput mesh in DOLFIN XDMF format.\n");
    fprintf(f, "\t\t\t\tThe XDMF file is written as XML to standard output,\n");
    fprintf(f, "\t\t\t\twhereas the mesh data is written to `FILE' in HDF5 format.\n");
    fprintf(f, "\n");
    fprintf(f, "  -q, --quiet\t\tdo not print output\n");
    fprintf(f, "  -v, --verbose\t\tbe more verbose\n");
    fprintf(f, "\n");
    fprintf(f, "  -h, --help\t\tdisplay this help and exit\n");
    fprintf(f, "  --version\t\tdisplay version information and exit\n");
    fprintf(f, "\n");
    fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * ‘program_options_print_version()’ prints version information.
 */
static void program_options_print_version(
    FILE * f)
{
    fprintf(f, "%s %s\n", program_name, program_version);
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * ‘parse_double()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * On success, ‘parse_double()’ returns ‘0’.  Otherwise, if the input
 * contained invalid characters, ‘parse_double()’ returns ‘EINVAL’.
 */
static int parse_double(
    const char * s,
    double * number)
{
    errno = 0;
    char * s_end;
    *number = strtod(s, &s_end);
    if ((errno == ERANGE && (*number == HUGE_VAL || *number == -HUGE_VAL)) ||
        (errno != 0 && number == 0)) {
        return errno;
    }
    if (s == s_end) return EINVAL;
    return 0;
}

/**
 * ‘parse_program_options()’ parses program options.
 */
static int parse_program_options(
    int argc,
    char ** argv,
    struct program_options * args,
    int * nargs)
{
    int err;
    *nargs = 0;
    (*nargs)++; argv++;

    while (*nargs < argc) {
        if (strcmp(argv[0], "--xmin") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->xmin);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--xmin=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--xmin="), &args->xmin);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--xmax") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->xmax);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--xmax=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--xmax="), &args->xmax);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--ymin") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->ymin);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--ymin=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--ymin="), &args->ymin);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--ymax") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->ymax);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--ymax=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--ymax="), &args->ymax);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--zmin") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->zmin);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--zmin=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--zmin="), &args->zmin);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--zmax") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->zmax);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--zmax=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--zmax="), &args->zmax);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--dx") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->dx);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--dx=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--dx="), &args->dx);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--dy") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->dy);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--dy=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--dy="), &args->dy);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--dz") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            err = parse_double(argv[0], &args->dz);
            if (err) return err;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--dz=") == argv[0]) {
            err = parse_double(argv[0]+strlen("--dz="), &args->dz);
            if (err) return err;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "--carp-pts") == 0) {
            args->meshformat = meshformat_carp_pts;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--carp-elem") == 0) {
            args->meshformat = meshformat_carp_elem;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--carp-lon") == 0) {
            args->meshformat = meshformat_carp_lon;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--tetgen-node") == 0) {
            args->meshformat = meshformat_tetgen_node;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--tetgen-ele") == 0) {
            args->meshformat = meshformat_tetgen_ele;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--dolfin-xdmf-ascii") == 0) {
            args->meshformat = meshformat_dolfin_xdmf_ascii;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--dolfin-xdmf-hdf5") == 0) {
            if (argc - *nargs < 2) return EINVAL;
            (*nargs)++; argv++;
            args->hdf5path = strdup(argv[0]);
            if (!args->hdf5path) return err;
            args->meshformat = meshformat_dolfin_xdmf_hdf5;
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--dolfin-xdmf-hdf5=") == argv[0]) {
            args->hdf5path = strdup(argv[0]+strlen("--dolfin-xdmf-hdf5="));
            if (!args->hdf5path) return err;
            args->meshformat = meshformat_dolfin_xdmf_hdf5;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = true;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++; continue;
        }

        /* if requested, print program help text */
        if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0) {
            program_options_free(args);
            program_options_print_help(stdout);
            exit(EXIT_SUCCESS);
        }

        /* if requested, print program version information */
        if (strcmp(argv[0], "--version") == 0) {
            program_options_free(args);
            program_options_print_version(stdout);
            exit(EXIT_SUCCESS);
        }

        /* stop parsing options after '--' */
        if (strcmp(argv[0], "--") == 0) {
            (*nargs)++; argv++;
            break;
        }

        /* unrecognised option */
        return EINVAL;
    }
    return 0;
}

static int64_t linear_rank_3(
    int64_t i0, int64_t i1, int64_t i2,
    int64_t n0, int64_t n1, int64_t n2)
{
    return i0*n1*n2 + i1*n2 + i2;
}

static int64_t linear_rank_4(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    int64_t n0, int64_t n1, int64_t n2, int64_t n3)
{
    return i0*n1*n2*n3 + i1*n2*n3 + i2*n3 + i3;
}

/**
 * ‘cuboid_mesh()’ creates a three-dimensional, tetrahedral mesh of a
 * cuboid.
 *
 * The extent of the cuboid is given by ‘[xmin,xmax]’ in the
 * x-direction, ‘[ymin,ymax]’ in the y-direction and ‘[zmin,zmax]’ in
 * the z-direction. Further, The spatial resolution in each direction
 * is given by ‘dx’, ‘dy’ and ‘dz’. Based on the given parameters, the
 * cuboid is divided into a number of uniform, rectangular cuboids
 * with side lengths ‘dx’, ‘dy’ and ‘dz’. Then, a tetrahedral mesh is
 * created by sub-dividing each rectangular cuboid into six
 * tetrahedra.
 *
 * On success, ‘cuboid_mesh()’ returns ‘0’, and ‘out_num_vertices’ and
 * ‘out_num_cells’ are set to the number of vertices and cells of the
 * mesh, respectively.  Moreover, a pointer to an array containing the
 * coordinates of each vertex is returned in ‘out_vertex_coordinates’,
 * and a pointer to an array containing the vertex indices of each
 * cell is returned in ‘out_cells’.  Otherwise, if an error occurs,
 * the return value is ‘-1’, and ‘errno’ is set to indicate the
 * appropriate error.
 */
static int cuboid_mesh(
    double xmin,
    double xmax,
    double dx,
    double ymin,
    double ymax,
    double dy,
    double zmin,
    double zmax,
    double dz,
    int64_t * out_num_vertices,
    int * out_num_coordinates_per_vertex,
    double ** out_vertex_coordinates,
    int64_t * out_num_cells,
    int * out_num_vertices_per_cell,
    int64_t ** out_cells,
    int * out_num_fibres_per_cell,
    int * out_num_coordinates_per_fibre,
    double ** out_fibres)
{
    int64_t num_vertices;
    double (*vertex_coordinates)[3];
    int64_t num_cells;
    int64_t (*cells)[4];

    /* Determine the number of cuboids along each direction. */
    int nx = ceil((xmax - xmin) / dx);
    int ny = ceil((ymax - ymin) / dy);
    int nz = ceil((zmax - zmin) / dz);
    if (nx <= 0 || ny <= 0 || nz <= 0)
        return EINVAL;

    int num_cells_per_unit = 6;
    int num_vertices_per_cell = 4;
    int num_coordinates_per_vertex = 3;
    if (__builtin_mul_overflow(nx+1, ny+1, &num_vertices) ||
        __builtin_mul_overflow(num_vertices, nz+1, &num_vertices))
    {
        return EOVERFLOW;
    }

    size_t vertex_coordinates_size;
    if (__builtin_mul_overflow(
            num_vertices, num_coordinates_per_vertex, &vertex_coordinates_size) ||
        __builtin_mul_overflow(
            vertex_coordinates_size, sizeof(double), &vertex_coordinates_size))
    {
        return EOVERFLOW;
    }

    vertex_coordinates = malloc(vertex_coordinates_size);
    if (!vertex_coordinates)
        return errno;

    #pragma omp parallel for
    for (int i0 = 0; i0 <= nx; i0++) {
        for (int i1 = 0; i1 <= ny; i1++) {
            for (int i2 = 0; i2 <= nz; i2++) {
                double x = xmin + (double)i0 * dx;
                if (x < xmin) x = xmin; else if (x > xmax) x = xmax;
                vertex_coordinates[linear_rank_3(i0,i1,i2,nx+1,ny+1,nz+1)][0] = x;
                double y = ymin + (double)i1 * dy;
                if (y < ymin) y = ymin; else if (y > ymax) y = ymax;
                vertex_coordinates[linear_rank_3(i0,i1,i2,nx+1,ny+1,nz+1)][1] = y;
                double z = zmin + (double)i2 * dz;
                if (z < zmin) z = zmin; else if (z > zmax) z = zmax;
                vertex_coordinates[linear_rank_3(i0,i1,i2,nx+1,ny+1,nz+1)][2] = z;
            }
        }
    }

    if (__builtin_mul_overflow(num_cells_per_unit, nx, &num_cells) ||
        __builtin_mul_overflow(num_cells, ny, &num_cells) ||
        __builtin_mul_overflow(num_cells, nz, &num_cells))
    {
        free(vertex_coordinates);
        return EOVERFLOW;
    }

    size_t cells_size;
    if (__builtin_mul_overflow(num_cells, num_vertices_per_cell, &cells_size) ||
        __builtin_mul_overflow(cells_size, sizeof(int64_t), &cells_size))
    {
        free(vertex_coordinates);
        return EOVERFLOW;
    }

    cells = malloc(cells_size);
    if (!cells) {
        free(vertex_coordinates);
        return errno;
    }

    #pragma omp parallel for
    for (int i0 = 0; i0 < nx; i0++) {
        for (int i1 = 0; i1 < ny; i1++) {
            for (int i2 = 0; i2 < nz; i2++) {
                int v[8][3] = {
                    {i0+0,i1+0,i2+0},
                    {i0+0,i1+0,i2+1},
                    {i0+0,i1+1,i2+0},
                    {i0+0,i1+1,i2+1},
                    {i0+1,i1+0,i2+0},
                    {i0+1,i1+0,i2+1},
                    {i0+1,i1+1,i2+0},
                    {i0+1,i1+1,i2+1}};
                int * cell_coords[6][4] = {
                    {v[0],v[1],v[3],v[7]},
                    {v[0],v[1],v[5],v[7]},
                    {v[0],v[2],v[3],v[7]},
                    {v[0],v[2],v[6],v[7]},
                    {v[0],v[4],v[5],v[7]},
                    {v[0],v[4],v[6],v[7]}};
                for (int j = 0; j < num_cells_per_unit; j++) {
                    for (int k = 0; k < num_vertices_per_cell; k++) {
                        cells[linear_rank_4(i0,i1,i2,j,nx,ny,nz,num_cells_per_unit)][k] =
                            linear_rank_3(
                                cell_coords[j][k][0],
                                cell_coords[j][k][1],
                                cell_coords[j][k][2],
                                nx+1, ny+1, nz+1);
                    }
                }
            }
        }
    }

    int num_fibres_per_cell = 1;
    int num_coordinates_per_fibre = 3;
    size_t fibres_size;
    if (__builtin_mul_overflow(num_cells, num_fibres_per_cell, &fibres_size) ||
        __builtin_mul_overflow(num_coordinates_per_fibre, fibres_size, &fibres_size) ||
        __builtin_mul_overflow(sizeof(double), fibres_size, &fibres_size))
    {
        free(cells);
        free(vertex_coordinates);
        return EOVERFLOW;
    }

    double (*fibres)[3] = malloc(fibres_size);
    if (!fibres) {
        free(cells);
        free(vertex_coordinates);
        return errno;
    }

    #pragma omp parallel for
    for (int i0 = 0; i0 < nx; i0++) {
        for (int i1 = 0; i1 < ny; i1++) {
            for (int i2 = 0; i2 < nz; i2++) {
                for (int j = 0; j < num_cells_per_unit; j++) {
                    size_t k = linear_rank_4(i0,i1,i2,j,nx,ny,nz,num_cells_per_unit);
                    fibres[k][0] = 1.0;
                    fibres[k][1] = 0.0;
                    fibres[k][2] = 0.0;
                }
            }
        }
    }

    *out_num_vertices = num_vertices;
    *out_num_coordinates_per_vertex = num_coordinates_per_vertex;
    *out_vertex_coordinates = (double *) vertex_coordinates;
    *out_num_cells = num_cells;
    *out_num_vertices_per_cell = num_vertices_per_cell;
    *out_cells = (int64_t *) cells;
    *out_num_fibres_per_cell = num_fibres_per_cell;
    *out_num_coordinates_per_fibre = num_coordinates_per_fibre;
    *out_fibres = (double *) fibres;
    return 0;
}

/**
 * ‘write_carp_pts()’ writes a set of nodes (i.e., coordinates of the
 * mesh vertices) to a stream in a format compatible with the Lynx
 * cardiac electrophysiology simulator.
 */
static int write_carp_pts(
    FILE * f,
    int64_t num_vertices,
    int num_coordinates_per_vertex,
    const double * vertex_coordinates)
{
    if (num_coordinates_per_vertex != 3) return EINVAL;

    fprintf(f, "%"PRId64"\n", num_vertices);

    size_t k = 0;
    for (int64_t i = 0; i < num_vertices; i++, k+=3) {
        fprintf(f, "%g %g %g\n",
                vertex_coordinates[k+0],
                vertex_coordinates[k+1],
                vertex_coordinates[k+2]);
    }
    return 0;
}

/**
 * ‘write_carp_elem()’ writes a set of elements (i.e., tetrahedral
 * mesh cells) to a stream in a format compatible with the Lynx
 * cardiac electrophysiology simulator.
 */
static int write_carp_elem(
    FILE * f,
    int64_t num_cells,
    int num_vertices_per_cell,
    const int64_t * cells)
{
    if (num_vertices_per_cell != 4) return EINVAL;

    fprintf(f, "%"PRId64"\n", num_cells);

    int region = 0;
    size_t k = 0;
    for (int64_t i = 0; i < num_cells; i++, k+=4) {
        fprintf(f, "Tt %"PRId64" %"PRId64" %"PRId64" %"PRId64" %d\n",
                cells[k+0], cells[k+1], cells[k+2], cells[k+3], region);
    }
    return 0;
}

/**
 * ‘write_carp_lon()’ writes a set of fibre directions (for a
 * transversely isotropic material model) to a stream in a format
 * compatible with the Lynx cardiac electrophysiology simulator.
 */
static int write_carp_lon(
    FILE * f,
    int64_t num_cells,
    int num_fibres_per_cell,
    int num_coordinates_per_fibre,
    const double * fibres)
{
    if (num_fibres_per_cell != 1) return EINVAL;
    if (num_coordinates_per_fibre != 3) return EINVAL;

    fprintf(f, "%d\n", num_fibres_per_cell);

    size_t k = 0;
    for (int64_t i = 0; i < num_cells; i++, k+=3) {
        fprintf(f, "%g %g %g\n",
                fibres[k+0], fibres[k+1], fibres[k+2]);
    }
    return 0;
}

/**
 * ‘write_tetgen_node()’ writes a set of nodes (i.e., coordinates of
 * the mesh vertices) to a stream in Tetgen ASCII format.
 */
static int write_tetgen_node(
    FILE * f,
    int64_t num_vertices,
    int num_coordinates_per_vertex,
    const double * vertex_coordinates)
{
    if (num_coordinates_per_vertex != 3) return EINVAL;

    fprintf(f, "%"PRId64" %d %d %d\n", num_vertices, num_coordinates_per_vertex, 0, 0);

    size_t k = 0;
    for (int64_t i = 0; i < num_vertices; i++, k+=3) {
        fprintf(f, "%"PRId64" %g %g %g\n",
                i,
                vertex_coordinates[k+0],
                vertex_coordinates[k+1],
                vertex_coordinates[k+2]);
    }
    return 0;
}

/**
 * ‘write_tetgen_ele()’ writes a set of elements (i.e., tetrahedral
 * mesh cells) to a stream in a Tetgen ASCII format.
 */
static int write_tetgen_ele(
    FILE * f,
    int64_t num_cells,
    int num_vertices_per_cell,
    const int64_t * cells)
{
    if (num_vertices_per_cell != 4) return EINVAL;

    fprintf(f, "%"PRId64" %d %d\n", num_cells, num_vertices_per_cell, 0);

    size_t k = 0;
    for (int64_t i = 0; i < num_cells; i++, k+=4) {
        fprintf(f, "%"PRId64" %"PRId64" %"PRId64" %"PRId64" %"PRId64"\n",
                i, cells[k+0], cells[k+1], cells[k+2], cells[k+3]);
    }
    return 0;
}

/**
 * ‘write_dolfin_xdmf_ascii()’ writes a mesh in DOLFIN XDMF ASCII
 * format.
 */
static int write_dolfin_xdmf_ascii(
    FILE * f,
    int64_t num_vertices,
    int num_coordinates_per_vertex,
    const double * vertex_coordinates,
    int64_t num_cells,
    int num_vertices_per_cell,
    const int64_t * cells)
{
    if (num_coordinates_per_vertex != 3) return EINVAL;
    if (num_vertices_per_cell != 4) return EINVAL;

    fprintf(f, "<Xdmf Version=\"3.0\">\n");
    fprintf(f, "  <Domain>\n");
    fprintf(f, "    <Grid Name=\"cuboid\">\n");
    fprintf(f, "      <Geometry GeometryType=\"XYZ\">\n");
    fprintf(f, "        <DataItem DataType=\"Float\" Dimensions=\"%"PRId64" %d\" Format=\"XML\" Precision=\"8\">\n",
            num_vertices, num_coordinates_per_vertex);
    size_t k = 0;
    for (int64_t i = 0; i < num_vertices; i++, k+=3) {
        fprintf(f, "%g %g %g\n",
                vertex_coordinates[k+0],
                vertex_coordinates[k+1],
                vertex_coordinates[k+2]);
    }
    fprintf(f, "        </DataItem>\n");
    fprintf(f, "      </Geometry>\n");

    fprintf(f, "      <Topology NumberOfElements=\"%"PRId64"\" TopologyType=\"Tetrahedron\">\n", num_cells);
    fprintf(f, "        <DataItem DataType=\"Int\" Dimensions=\"%"PRId64" %d\" Format=\"XML\" Precision=\"8\">\n",
            num_cells, num_vertices_per_cell);
    k = 0;
    for (int64_t i = 0; i < num_cells; i++, k+=4) {
        fprintf(f, "%"PRId64" %"PRId64" %"PRId64" %"PRId64"\n",
                cells[k+0], cells[k+1], cells[k+2], cells[k+3]);
    }
    fprintf(f, "        </DataItem>\n");
    fprintf(f, "      </Topology>\n");
    fprintf(f, "    </Grid>\n");
    fprintf(f, "  </Domain>\n");
    fprintf(f, "</Xdmf>\n");
    return 0;
}

#ifdef HAVE_HDF5
/**
 * ‘write_dolfin_xdmf_HDF5()’ writes a mesh in DOLFIN XDMF HDF5
 * format.
 */
static int write_dolfin_xdmf_hdf5(
    FILE * f,
    const char * hdf5path,
    int64_t num_vertices,
    int num_coordinates_per_vertex,
    const double * vertex_coordinates,
    int64_t num_cells,
    int num_vertices_per_cell,
    const int64_t * cells)
{
    if (num_coordinates_per_vertex != 3) return EINVAL;
    if (num_vertices_per_cell != 4) return EINVAL;

    herr_t status;
    hid_t file_id = H5Fcreate(hdf5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    const char * geometry_group = "/geometry";
    const char * geometry_dset = "/geometry/x";
    {
        hid_t group_id = H5Gcreate(
            file_id, geometry_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {num_vertices, num_coordinates_per_vertex};
        hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dataset_id = H5Dcreate2(
            file_id, geometry_dset, H5T_NATIVE_DOUBLE,
            dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(
            dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            vertex_coordinates);
        status = H5Dclose(dataset_id);
        status = H5Sclose(dataspace_id);
        status = H5Gclose (group_id);
    }
    const char * topology_group = "/topology";
    const char * topology_dset = "/topology/cells";
    {
        hid_t group_id = H5Gcreate(
            file_id, topology_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {num_cells, num_vertices_per_cell};
        hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dataset_id = H5Dcreate2(
            file_id, topology_dset, H5T_NATIVE_INT64,
            dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(
            dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, cells);
        status = H5Dclose(dataset_id);
        status = H5Sclose(dataspace_id);
        status = H5Gclose (group_id);
    }
    status = H5Fclose(file_id);

    fprintf(f, "<Xdmf Version=\"3.0\">\n");
    fprintf(f, "  <Domain>\n");
    fprintf(f, "    <Grid Name=\"cuboid\">\n");
    fprintf(f, "      <Geometry GeometryType=\"XYZ\">\n");
    fprintf(f, "        <DataItem DataType=\"Float\" Dimensions=\"%"PRId64" %d\" Format=\"HDF\" Precision=\"%ld\">\n",
            num_vertices, num_coordinates_per_vertex, sizeof(*vertex_coordinates));
    fprintf(f, "          %s:%s\n", hdf5path, geometry_dset);
    fprintf(f, "        </DataItem>\n");
    fprintf(f, "      </Geometry>\n");

    fprintf(f, "      <Topology NumberOfElements=\"%"PRId64"\" TopologyType=\"Tetrahedron\">\n", num_cells);
    fprintf(f, "        <DataItem DataType=\"Int\" Dimensions=\"%"PRId64" %d\" Format=\"HDF\" Precision=\"%ld\">\n",
            num_cells, num_vertices_per_cell, sizeof(*cells));
    fprintf(f, "          %s:%s\n", hdf5path, topology_dset);
    fprintf(f, "        </DataItem>\n");
    fprintf(f, "      </Topology>\n");
    fprintf(f, "    </Grid>\n");
    fprintf(f, "  </Domain>\n");
    fprintf(f, "</Xdmf>\n");
    return 0;
}
#else
/**
 * ‘write_dolfin_xdmf_HDF5()’ writes a mesh in DOLFIN XDMF HDF5
 * format.
 */
static int write_dolfin_xdmf_hdf5(
    FILE * f,
    const char * hdf5path,
    int num_vertices,
    int num_coordinates_per_vertex,
    const double * vertex_coordinates,
    int64_t num_cells,
    int num_vertices_per_cell,
    const int64_t * cells)
{
    return ENOTSUP;
}
#endif

/**
 * ‘timespec_duration()’ is the duration, in seconds, elapsed between
 * two given time points.
 */
static double timespec_duration(
    struct timespec t0,
    struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static int write_mesh(
    FILE * f,
    enum meshformat meshformat,
    int64_t num_vertices,
    int num_coordinates_per_vertex,
    const double * vertex_coordinates,
    int64_t num_cells,
    int num_vertices_per_cell,
    const int64_t * cells,
    int num_fibres_per_cell,
    int num_coordinates_per_fibre,
    const double * fibres,
    const char * hdf5path,
    int verbose)
{
    if (meshformat == meshformat_carp_pts) {
        return write_carp_pts(
            f, num_vertices, num_coordinates_per_vertex, vertex_coordinates);
    } else if (meshformat == meshformat_carp_elem) {
        return write_carp_elem(
            f, num_cells, num_vertices_per_cell, cells);
    } else if (meshformat == meshformat_carp_lon) {
        return write_carp_lon(
            f, num_cells, num_fibres_per_cell,
            num_coordinates_per_fibre, fibres);
    } else if (meshformat == meshformat_tetgen_node) {
        return write_tetgen_node(
            f, num_vertices, num_coordinates_per_vertex, vertex_coordinates);
    } else if (meshformat == meshformat_tetgen_ele) {
        return write_tetgen_ele(
            f, num_cells, num_vertices_per_cell, cells);
    } else if (meshformat == meshformat_dolfin_xdmf_ascii) {
        return write_dolfin_xdmf_ascii(
            f, num_vertices, num_coordinates_per_vertex, vertex_coordinates,
            num_cells, num_vertices_per_cell, cells);
    } else if (meshformat == meshformat_dolfin_xdmf_hdf5) {
        return write_dolfin_xdmf_hdf5(
            f, hdf5path,
            num_vertices, num_coordinates_per_vertex, vertex_coordinates,
            num_cells, num_vertices_per_cell, cells);
    } else { return EINVAL; }
}

/**
 * ‘main()’.
 */
int main(int argc, char *argv[])
{
#ifndef _GNU_SOURCE
    /* set program invocation name */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);
#endif

    /* set default program options */
    struct program_options args;
    int err = program_options_init(&args);
    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        return EXIT_FAILURE;
    }

    /* parse program options */
    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* 1. generate the cuboid mesh */
    struct timespec t0, t1;
    if (args.verbose > 0) {
        fprintf(stderr, "generating cuboid mesh of the region "
                "[%g,%g]x[%g,%g]x[%g,%g] with resolution %gx%gx%g: ",
                args.xmin, args.xmax,
                args.ymin, args.ymax,
                args.zmin, args.zmax,
                args.dx, args.dy, args.dz);
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t num_vertices;
    int num_coordinates_per_vertex;
    double * vertex_coordinates;
    int64_t num_cells;
    int num_vertices_per_cell;
    int64_t * cells;
    int num_fibres_per_cell;
    int num_coordinates_per_fibre;
    double * fibres;
    err = cuboid_mesh(
        args.xmin, args.xmax, args.dx,
        args.ymin, args.ymax, args.dy,
        args.zmin, args.zmax, args.dz,
        &num_vertices,
        &num_coordinates_per_vertex,
        &vertex_coordinates,
        &num_cells,
        &num_vertices_per_cell,
        &cells,
        &num_fibres_per_cell,
        &num_coordinates_per_fibre,
        &fibres);
    if (err) {
        if (args.verbose > 0)
            fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%.6f seconds %"PRId64" vertices %"PRId64" cells\n",
                timespec_duration(t0, t1), num_vertices, num_cells);
    }

    /* 2. output mesh */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(stderr, "writing mesh: ");
            fflush(stderr);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        err = write_mesh(
            stdout, args.meshformat,
            num_vertices, num_coordinates_per_vertex, vertex_coordinates,
            num_cells, num_vertices_per_cell, cells,
            num_fibres_per_cell, num_coordinates_per_fibre, fibres,
            args.hdf5path, args.verbose);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
            free(fibres);
            free(cells);
            free(vertex_coordinates);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%.6f seconds\n",
                    timespec_duration(t0, t1));
        }
    }

    /* 3. clean up */
    free(fibres);
    free(cells);
    free(vertex_coordinates);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
