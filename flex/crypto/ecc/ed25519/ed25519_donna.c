
#define HAVE_UINT128
#define ED25519_64BIT
//#define ED25519_SSE2
//#define ED25519_32BIT
//#define ED25519_FORCE_32BIT

#include <Python.h>
#include <ed25519-donna.h>


ge25519 g;
unsigned char g_compressed[32];

static void to_hex_swap(unsigned char *, unsigned char *);
static void from_hex_swap(unsigned char *, unsigned char *);
static void ge25519_get_hex_xy(ge25519 *, unsigned char *, unsigned char *);
static void ge25519_set_hex_xy(ge25519 *, unsigned char *, unsigned char *);
static void ge25519_print(ge25519 *, char *);
static void ge25519_double_scalarmult_vartime_patched(ge25519 *, const ge25519 *, const bignum256modm, const bignum256modm);
static int ge25519_unpack(ge25519 *, const unsigned char *);

static PyObject* mul(PyObject* self, PyObject* args)
{

	PyObject *value, *list;
	if (!PyArg_ParseTuple(args, "OO", &list, &value)) {
		return NULL;
	}

	if (!PyLong_Check(value)) {
		PyErr_SetString(PyExc_TypeError, "expected int or long");
		return NULL;
	}

	if (!PyList_Check(list)) {
		PyErr_SetString(PyExc_TypeError, "expected list");
		return NULL;
	}

        // load point
	PyObject *valuex, *valuey;
	valuex = PyList_GetItem(list, 0);
	valuey = PyList_GetItem(list, 1);

	// check if point at infinity received
	if (valuex==Py_None) {
		return Py_BuildValue("[O,O]", Py_None,Py_None);
	}

        // convert to hex string
//    PyObject* fmt = PyBytes_FromString("%064x");
    PyObject* fmt = PyUnicode_FromString("%064x");
	PyObject* pyhex = PyUnicode_Format(fmt, value);
	char* hex = PyUnicode_AsUTF8(pyhex);
//	char* hex = PyBytes_AsString(pyhex);
	PyObject* pyhexPx = PyUnicode_Format(fmt, valuex);
	char* hexPx = PyUnicode_AsUTF8(pyhexPx);
//	char* hexPx = PyBytes_AsString(pyhexPx);
	PyObject* pyhexPy = PyUnicode_Format(fmt, valuey);
	char* hexPy = PyUnicode_AsUTF8(pyhexPy);
//	char* hexPy = PyBytes_AsString(pyhexPy);

	// load point
	ge25519 p;
	ge25519_set_hex_xy(&p, hexPx, hexPy);

	// load scalar
	char buf[32];
        from_hex_swap(buf, hex);
	bignum256modm a;
	expand256_modm(a, buf, 32);

	Py_DECREF(fmt);
	Py_DECREF(pyhex);
	Py_DECREF(pyhexPx);
	Py_DECREF(pyhexPy);


	// use optimization if multiplication is performed on the base point
	unsigned char p_compressed[32];
	ge25519_pack(p_compressed, &p);

	if (memcmp(g_compressed, p_compressed, 32)==0) {

		// perform scalar multiplication of G
		ge25519_scalarmult_base_niels(&p, ge25519_niels_base_multiples, a);

	} else {

		// zero
		unsigned char charzero[32] = {0};
		bignum256modm zero;
		expand256_modm(zero, charzero, 32);

		ge25519_double_scalarmult_vartime_patched(&p, &p, a, zero);
	}

	// check if the result is point at infinity
	// edwards curves do not have a point at infinity

	// convert x and y to hex
	unsigned char cx[65];
	unsigned char cy[65];
	ge25519_get_hex_xy(&p, cx, cy);

	// convert hex string of x, y to python long
	PyObject* xr = PyLong_FromString(cx, NULL, 16);
	PyObject* yr = PyLong_FromString(cy, NULL, 16);

	return Py_BuildValue("[NN]", xr, yr);

}

static PyObject* add(PyObject* self, PyObject* args)
{

	PyObject *list1, *list2;
	if (!PyArg_ParseTuple(args, "OO", &list1, &list2)) {
		return NULL;
	}

	if (!PyList_Check(list1) || !PyList_Check(list2)) {
		PyErr_SetString(PyExc_TypeError, "expected list");
		return NULL;
	}

	PyObject *oAx, *oAy;
	oAx = PyList_GetItem(list1, 0);
	oAy = PyList_GetItem(list1, 1);

	PyObject *oBx, *oBy;
	oBx = PyList_GetItem(list2, 0);
	oBy = PyList_GetItem(list2, 1);

	// check if point at infinity received
	if (oAx==Py_None) {
		return Py_BuildValue("O", list2);
	}
	if (oBx==Py_None) {
		return Py_BuildValue("O", list1);
	}


	// convert to hex string
	PyObject* fmt = PyBytes_FromString("%064x");
	PyObject* pyhexAx = PyUnicode_Format(fmt, oAx);
	char* hexAx = PyUnicode_AsUTF8(pyhexAx);
//	char* hexAx = PyBytes_AsString(pyhexAx);
	PyObject* pyhexAy = PyUnicode_Format(fmt, oAy);
	char* hexAy = PyUnicode_AsUTF8(pyhexAy);
//	char* hexAy = PyBytes_AsString(pyhexAy);
	PyObject* pyhexBx = PyUnicode_Format(fmt, oBx);
	char* hexBx = PyUnicode_AsUTF8(pyhexBx);
//	char* hexBx = PyBytes_AsString(pyhexBx);
	PyObject* pyhexBy = PyUnicode_Format(fmt, oBy);
	char* hexBy = PyUnicode_AsUTF8(pyhexBy);
//	char* hexBy = PyBytes_AsString(pyhexBy);

	// load points
	ge25519 a, b;
	ge25519_set_hex_xy(&a, hexAx, hexAy);
	ge25519_set_hex_xy(&b, hexBx, hexBy);

	Py_DECREF(fmt);
	Py_DECREF(pyhexAx);
	Py_DECREF(pyhexAy);
	Py_DECREF(pyhexBx);
	Py_DECREF(pyhexBy);

	ge25519 r;

	// perform addition
	ge25519_add(&r, &a, &b);

	// check if the result is point at infinity
	// edwards curves do not have a point at infinity

	// convert x and y to hex
	unsigned char cx[65];
	unsigned char cy[65];
	ge25519_get_hex_xy(&r, cx, cy);

	// convert hex string of x, y to python long
	PyObject* xr = PyLong_FromString(cx, NULL, 16);
	PyObject* yr = PyLong_FromString(cy, NULL, 16);

	return Py_BuildValue("[NN]", xr, yr);

}


static PyObject* inv(PyObject* self, PyObject* args)
{

	PyObject *list;
	if (!PyArg_ParseTuple(args, "O", &list)) {
		return NULL;
	}

	if (!PyList_Check(list)) {
		PyErr_SetString(PyExc_TypeError, "expected list");
		return NULL;
	}

	PyObject *Px, *Py;
	Px = PyList_GetItem(list, 0);
	Py = PyList_GetItem(list, 1);

	// check if point at infinity received
	// edwards curves do not have a point at infinity

	// convert to hex string
	PyObject* fmt = PyBytes_FromString("%064x");
	PyObject* pyhexPx = PyUnicode_Format(fmt, Px);
	char* hexPx = PyUnicode_AsUTF8(pyhexPx);
//	char* hexPx = PyBytes_AsString(pyhexPx);
	PyObject* pyhexPy = PyUnicode_Format(fmt, Py);
	char* hexPy = PyUnicode_AsUTF8(pyhexPy);
//	char* hexPy = PyBytes_AsString(pyhexPy);

	// load point
	ge25519 p;
	ge25519_set_hex_xy(&p, hexPx, hexPy);

	Py_DECREF(fmt);
	Py_DECREF(pyhexPx);
	Py_DECREF(pyhexPy);

	// perform inverting
	unsigned char buf[32];
	ge25519_pack(buf, &p);
	ge25519_unpack_negative_vartime(&p, buf);

	// convert x and y to hex
	unsigned char cx[65];
	unsigned char cy[65];
	ge25519_get_hex_xy(&p, cx, cy);

	// convert hex string of x, y to python long
	PyObject* xr = PyLong_FromString(cx, NULL, 16);
	PyObject* yr = PyLong_FromString(cy, NULL, 16);

	return Py_BuildValue("[NN]", xr, yr);
}

static PyObject* compress(PyObject* self, PyObject* args)
{

	PyObject *list;
	if (!PyArg_ParseTuple(args, "O", &list)) {
		return NULL;
	}

	if (!PyList_Check(list)) {
		PyErr_SetString(PyExc_TypeError, "expected list");
		return NULL;
	}

	PyObject *Px, *Py;
	Px = PyList_GetItem(list, 0);
	Py = PyList_GetItem(list, 1);

	// check if point at infinity received
	// edwards curves do not have a point at infinity

	// convert to hex string
	PyObject* fmt = PyBytes_FromString("%064x");
	PyObject* pyhexPx = PyUnicode_Format(fmt, Px);
	char* hexPx = PyUnicode_AsUTF8(pyhexPx);
//	char* hexPx = PyBytes_AsString(pyhexPx);
	PyObject* pyhexPy = PyUnicode_Format(fmt, Py);
	char* hexPy = PyUnicode_AsUTF8(pyhexPy);
//	char* hexPy = PyBytes_AsString(pyhexPy);

	// load point
	ge25519 p;
	ge25519_set_hex_xy(&p, hexPx, hexPy);

	Py_DECREF(fmt);
	Py_DECREF(pyhexPx);
	Py_DECREF(pyhexPy);


	// perform compression
        unsigned char x[32];
        unsigned char y[32];
        bignum25519 tx, ty, zi;
        curve25519_recip(zi, p.z);
        curve25519_mul(tx, p.x, zi);
        curve25519_mul(ty, p.y, zi);
        curve25519_contract(x, tx);
        curve25519_contract(y, ty);

	// encode sign of Y coordinate
        unsigned char pub[33];
	pub[0] = 0x02;
	if (y[0] & 1) {
		pub[0] = 0x03;
	}

	// swap to big endian order
	for (int i=0; i < 32; i++) {
		pub[i+1] = x[31-i];
	}

	// convert char array to python string object
	PyObject* s = PyBytes_FromStringAndSize(pub,33);

	return Py_BuildValue("N", s);

}


static PyObject* decompress(PyObject* self, PyObject* args)
{

	PyObject *pubs;
	if (!PyArg_ParseTuple(args, "O", &pubs)) {
		return NULL;
	}

	if (!PyBytes_Check(pubs)) {
		PyErr_SetString(PyExc_TypeError, "string expected");
		return NULL;
	}

	char* pub = PyBytes_AsString(pubs);

	// check if point at infinity received
	// edwards curves do not have a point at infinity

	// swap to big endian order
	unsigned char y[32];
	for (int i=0; i < 32; i++) {
		y[i] = pub[32-i];
	}
	y[31] ^= (pub[0] & 1) << 7;

	// perform decompression
	ge25519 p;
	ge25519_unpack(&p, y);

	// convert x and y to hex
	unsigned char cx[65];
	unsigned char cy[65];
	ge25519_get_hex_xy(&p, cx, cy);

	// convert hex string of x, y to python long
	PyObject* xr = PyLong_FromString(cx, NULL, 16);
	PyObject* yr = PyLong_FromString(cy, NULL, 16);

	return Py_BuildValue("[NN]", xr, yr);
}


static PyObject* valid(PyObject* self, PyObject* args)
{

	PyObject *list;
	if (!PyArg_ParseTuple(args, "O", &list)) {
		return NULL;
	}

	if (!PyList_Check(list)) {
		PyErr_SetString(PyExc_TypeError, "expected list");
		return NULL;
	}

	PyObject *Px, *Py;
	Px = PyList_GetItem(list, 0);
	Py = PyList_GetItem(list, 1);

	// check if point at infinity received
	// edwards curves do not have a point at infinity

	// convert to hex string
	PyObject* fmt = PyBytes_FromString("%064x");
	PyObject* pyhexPx = PyUnicode_Format(fmt, Px);
	char* hexPx = PyUnicode_AsUTF8(pyhexPx);
//	char* hexPx = PyBytes_AsString(pyhexPx);
	PyObject* pyhexPy = PyUnicode_Format(fmt, Py);
	char* hexPy = PyUnicode_AsUTF8(pyhexPy);
//	char* hexPy = PyBytes_AsString(pyhexPy);

	// convert point to little endian
	unsigned char x[32];
	unsigned char y[32];
	from_hex_swap(x,hexPx);
	from_hex_swap(y,hexPy);

	Py_DECREF(fmt);
	Py_DECREF(pyhexPx);
	Py_DECREF(pyhexPy);

	// validate point
	// return (-x*x + y*y - 1 - self.d*x*x*y*y) % self.p == 0
	ge25519 r;
	bignum25519 numl, numr, numy, numx;
	static const bignum25519 one = {1};

	curve25519_expand(r.x, x);
	curve25519_expand(r.y, y);
	curve25519_square(r.x, r.x); /* x = x^2 */
	curve25519_square(r.y, r.y); /* y = y^2 */
	curve25519_mul(numr, r.x, r.y); /* numr = x*x*y*y */
	curve25519_mul(numr, numr, ge25519_ecd); /* numr = d*x*x*y*y */
	
	curve25519_neg(numl, r.x); /* numl = -x*x */
	curve25519_add_reduce(numl, numl, r.y); /* numl = -x*x + y*y */
	curve25519_sub_reduce(numl, numl, one); /* numl = -x*x + y*y -1 */
	curve25519_sub_reduce(numl, numl, numr);  /* numl = -x*x + y*y -1 - numr */
	
	static const unsigned char zero[32] = {0};
	unsigned char check[32];
	curve25519_contract(check, numl);

	if (memcmp(check, zero, 32)==0) {
		return Py_BuildValue("i", 1);
	}

	return Py_BuildValue("i", 0);

}

static PyMethodDef Methods[] =
{
	{"mul", mul, METH_VARARGS, "Perform point multiplication"},
	{"add", add, METH_VARARGS, "Perform point addition"},
	{"inv", inv, METH_VARARGS, "Perform point inversion"},
	{"compress", compress, METH_VARARGS, "Compress point"},
	{"decompress", decompress, METH_VARARGS, "Decompress point"},
	{"valid", valid, METH_VARARGS, "Check whether the point is on curve"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef ed25519_donna_module =
{
    PyModuleDef_HEAD_INIT,
    "ed25519_donna",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_ed25519_donna(void)
{

	PyObject *module;
	module = PyModule_Create(&ed25519_donna_module);
	if (module == NULL)
	    return NULL;

	// export base point g
	unsigned char charone[32] = {1};
	bignum256modm one;
	expand256_modm(one, charone, 32);
	ge25519_scalarmult_base_niels(&g, ge25519_niels_base_multiples, one);
	ge25519_pack(g_compressed, &g);

	// convert x and y to hex
        unsigned char hexgx[65];
        unsigned char hexgy[65];
	ge25519_get_hex_xy(&g, hexgx, hexgy);

	// convert hex string of x, y to python long
	PyObject* xr = PyLong_FromString(hexgx, NULL, 16);
	PyObject* yr = PyLong_FromString(hexgy, NULL, 16);

	// export base point G as module variable "g"
	PyModule_AddObject(module, "g", Py_BuildValue("[NN]", xr, yr));


	// export group order n
	unsigned char m[32];
	contract256_modm(m, modm_m);

	// convert n to hex
	char cn[65];
	to_hex_swap(m, cn);

	// convert hex string of n to python long
	PyObject* n = PyLong_FromString(cn, NULL, 16);

	// export curve order as module variable "n"
	PyModule_AddObject(module, "n", Py_BuildValue("N", n));

	return module;

}

static void ge25519_set_hex_xy(ge25519* p, unsigned char* hexx, unsigned char* hexy) {

        
	unsigned char x[32];
	unsigned char y[32];
	from_hex_swap(x,hexx);
	from_hex_swap(y,hexy);
	y[31] ^= (((x[0] & 1) ^ 1) << 7);
	ge25519_unpack_negative_vartime(p, y);
}

static void ge25519_get_hex_xy(ge25519* p, unsigned char* hexx, unsigned char* hexy) {

	bignum25519 tx, ty, zi;
	unsigned char x[32];
	unsigned char y[32];
	curve25519_recip(zi, p->z);
	curve25519_mul(tx, p->x, zi);
	curve25519_mul(ty, p->y, zi);
	curve25519_contract(x, tx);
	curve25519_contract(y, ty);

	to_hex_swap(x,hexx);
	to_hex_swap(y,hexy);

}

static void to_hex_swap(unsigned char *r32, unsigned char *r64) {

	int i;
	for (i=0; i<32; i++) {
		/* Hex character table. */
		static const char *c = "0123456789ABCDEF";
		r64[62-2*i] = c[(r32[i] >> 4) & 0xF];
		r64[62-2*i+1] = c[(r32[i]) & 0xF];
	}
	r64[64]=0;
}

static void from_hex_swap(unsigned char *r32, unsigned char *r64) {
	int i;
	/* Byte to hex value table. */
	static const int cvt[256] = {0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 1, 2, 3, 4, 5, 6,7,8,9,0,0,0,0,0,0,
				0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,
				0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0};
	for (i=0; i<32; i++) {
		r32[31-i] = (cvt[(unsigned char)r64[2*i]] << 4) + cvt[(unsigned char)r64[2*i+1]];
	}
}


// for debugging
static void ge25519_print(ge25519* r, char* msg) {
        unsigned char res[33];

        printf("\n");
        curve25519_contract(res, r->x);
        printf("        x=");
        for (int i=0; i<32; i++) {
                printf("%02x", (unsigned)(unsigned char)res[i]);
        }
        printf("\n");

        curve25519_contract(res, r->y);
        printf("        y=");
        for (int i=0; i<32; i++) {
                printf("%02x", (unsigned)(unsigned char)res[i]);
        }
        printf("\n");

        curve25519_contract(res, r->z);
        printf("        z=");
        for (int i=0; i<32; i++) {
                printf("%02x", (unsigned)(unsigned char)res[i]);
        }
        printf("\n");

        curve25519_contract(res, r->t);
        printf("        t=");
        for (int i=0; i<32; i++) {
                printf("%02x", (unsigned)(unsigned char)res[i]);
        }
        printf("\n");

}


// patched version of ge25519_double_scalarmult_vartime() (see https://github.com/floodyberry/ed25519-donna/issues/31)
static void ge25519_double_scalarmult_vartime_patched(ge25519 *r, const ge25519 *p1, const bignum256modm s1, const bignum256modm s2) {
	signed char slide1[256], slide2[256];
	ge25519_pniels pre1[S1_TABLE_SIZE];
	ge25519 d1;
	ge25519_p1p1 t;
	int32_t i;

	contract256_slidingwindow_modm(slide1, s1, S1_SWINDOWSIZE);
	contract256_slidingwindow_modm(slide2, s2, S2_SWINDOWSIZE);

	ge25519_double(&d1, p1);
	ge25519_full_to_pniels(pre1, p1);
	for (i = 0; i < S1_TABLE_SIZE - 1; i++)
		ge25519_pnielsadd(&pre1[i+1], &d1, &pre1[i]);

	/* set neutral */
	memset(r, 0, sizeof(ge25519));
	r->y[0] = 1;
	r->z[0] = 1;

	i = 255;
	while ((i >= 0) && !(slide1[i] | slide2[i]))
		i--;

	for (; i >= 0; i--) {
		ge25519_double_p1p1(&t, r);

		if (slide1[i]) {
			ge25519_p1p1_to_full(r, &t);
			ge25519_pnielsadd_p1p1(&t, r, &pre1[abs(slide1[i]) / 2], (unsigned char)slide1[i] >> 7);
		}

		if (slide2[i]) {
			ge25519_p1p1_to_full(r, &t);
			ge25519_nielsadd2_p1p1(&t, r, &ge25519_niels_sliding_multiples[abs(slide2[i]) / 2], (unsigned char)slide2[i] >> 7);
		}

		ge25519_p1p1_to_partial(r, &t);
	}
	ge25519_p1p1_to_full(r, &t);
}

// unpack y coordinate
static int ge25519_unpack(ge25519 *r, const unsigned char p[32]) {
	static const unsigned char zero[32] = {0};
	static const bignum25519 one = {1};
	unsigned char parity = p[31] >> 7;
	unsigned char check[32];
	bignum25519 t, root, num, den, d3;

	// find y = sqrt((x^2+1) / (1-dx^2))
	curve25519_expand(r->x, p);
	curve25519_copy(r->z, one);
	curve25519_square(num, r->x); /* y = x^2 */
	curve25519_mul(den, num, ge25519_ecd); /* den = dx^2 */
	curve25519_add(num, num, r->z); /* y = x^2 + 1 */
	curve25519_sub_reduce(den, r->z, den); /* den = 1 - dx^2 */

	/* Computation of sqrt(num/den) */
	/* 1.: computation of num^((p-5)/8)*den^((7p-35)/8) = (num*den^7)^((p-5)/8) */
	curve25519_square(t, den);
	curve25519_mul(d3, t, den);
	curve25519_square(r->y, d3);
	curve25519_mul(r->y, r->y, den);
	curve25519_mul(r->y, r->y, num);
	curve25519_pow_two252m3(r->y, r->y);

	/* 2. computation of r->y = num * den^3 * (num*den^7)^((p-5)/8) */
	curve25519_mul(r->y, r->y, d3);
	curve25519_mul(r->y, r->y, num);

	/* 3. Check if either of the roots works: */
	curve25519_square(t, r->y);
	curve25519_mul(t, t, den);
	curve25519_sub_reduce(root, t, num);
	curve25519_contract(check, root);
	if (!ed25519_verify(check, zero, 32)) {
		curve25519_add_reduce(t, t, num);
		curve25519_contract(check, t);
		if (!ed25519_verify(check, zero, 32))
			return 0;
		curve25519_mul(r->y, r->y, ge25519_sqrtneg1);
	}

	curve25519_contract(check, r->y);
	if ((check[0] & 1) != parity) {
		curve25519_copy(t, r->y);
		curve25519_neg(r->y, t);
	}
	curve25519_mul(r->t, r->x, r->y);
	return 1;
}
