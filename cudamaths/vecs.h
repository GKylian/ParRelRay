#pragma once

//#include "cuda_runtime.h"



#include "if_interop.cuh"
#include "transform.cuh"

//#include <stdio.h>
//#include <math.h>
//#include <vector>

#define NEWVECS


#ifndef NEWVECS

#pragma region floats
	/*__host__ __device__*/ struct f2 {
		float x, y;
		f2(float _x, float _y) { x = _x; y = _y; }
		f2() { x = 0; y = 0; }

		float norm() {			return sqrtf(x*x+y*y);					}
		float norm2() {			return x*x+y*y+y;						}
		void normalize() { float n = norm(); x /= n; y /= n;			}
		void cinv() {			x = 1.0f/x; y = 1.0f/y;					}
	};
	typedef /*__host__ __device__*/ struct f2 f2;

	/*__host__ __device__*/ struct f3 {
		float x, y, z;
		f3(float _x, float _y, float _z) { x = _x; y = _y; z = _z; }
		f3() { x = 0; y = 0; z = 0; }
		f2 xy() { return f2(x, y); }   f2 xz() { return f2(x, z); }   f2 yz() { return f2(y, z); }

		float norm() {			return sqrtf(x*x+y*y+z*z);					}
		float norm2() {			return x*x+y*y+z*z;							}
		void normalize() {		float n = norm(); x /= n; y /= n; z /= n;	}
		void cinv() {			x = 1.0f/x; y = 1.0f/y; z = 1.0f/z;			}
	};
	typedef /*__host__ __device__*/ struct f3 f3;

	/*__host__ __device__*/ struct f4 {
		float x, y, z, w;
		f4(float _x, float _y, float _z, float _w) { x = _x; y = _y; z = _z; w = _w; }
		f4() { x = 0; y = 0; z = 0; w = 0; }
		f2 xy() { return f2(x, y); }   f2 xz() { return f2(x, z); }   f2 yz() { return f2(y, z); }
		f3 xyz() { return f3(x, y, z); }

		float norm() {			return sqrtf(x*x+y*y+z*z+w*w);							}
		float norm2() {			return x*x+y*y+z*z+w*w;									}
		void normalize() {		float n = norm(); x /= n; y /= n; z /= n; w /= n;		}
		void cinv() {			x = 1.0f/x; y = 1.0f/y; z = 1.0f/z; w = 1.0f/w;			}
	};
	typedef /*__host__ __device__*/ struct f4 f4;
#pragma endregion

#pragma region ints
	/*__host__ __device__*/ struct i2 {
		int x, y;
		i2(int _x, int _y) { x = _x; y = _y; }
		i2() { x = 0; y = 0; }

		float norm() {			return sqrtf(x*x+y*y);					}
		int norm2() {			return x*x+y*y+y;						}
	};
	typedef /*__host__ __device__*/ struct i2 i2;

	/*__host__ __device__*/ struct i3 {
		int x, y, z;
		i3(int _x, int _y, int _z) { x = _x; y = _y; z = _z; }
		i3() { x = 0; y = 0; z = 0; }
		i2 xy() { return i2(x, y); }   i2 xz() { return i2(x, z); }   i2 yz() { return i2(y, z); }

		float norm() {			return sqrtf(x*x+y*y+z*z);					}
		int norm2() {			return x*x+y*y+z*z;							}
	};
	typedef /*__host__ __device__*/ struct i3 i3;

	/*__host__ __device__*/ struct i4 {
		int x, y, z, w;
		i4(int _x, int _y, int _z, int _w) { x = _x; y = _y; z = _z; w = _w; }
		i4() { x = 0; y = 0; z = 0; w = 0; }
		i2 xy() { return i2(x, y); }   i2 xz() { return i2(x, z); }   i2 yz() { return i2(y, z); }
		i3 xyz() { return i3(x, y, z); }

		float norm() { return sqrtf(x*x+y*y+z*z+w*w); }
		int norm2() { return x*x+y*y+z*z+w*w; }
	};
	typedef /*__host__ __device__*/ struct i4 i4;
#pragma endregion


#pragma region matrices
	/*__host__ __device__*/ struct mat2 {
		float a[4];
		mat2() {			for (int i = 0; i < 4; i++) a[i] = 0;			}
		mat2(float _a[4]) { for (int i = 0; i < 4; i++) a[i] = _a[i];		}
		mat2(f2 _a, f2 _b) { a[0] = _a.x; a[1] = _b.x; a[2] = _a.y; a[3] = _b.y; } //Vectors to COLUMNS of the matrix

		/*__host__ __device__*/ inline f2 operator[](int i) { if (i >= 2) return f2(); return f2(a[0+i], a[2+i]); }

		void set(const mat2 &_a) { for (int i = 0; i < 4; i++) a[i] = _a.a[i]; }
		void cinv() {		for (int i = 0; i < 4; i++) a[i] = 1.0f/a[i];	}
		void inv() {
			mat2 mat; float det = a[0]*a[3]-a[2]*a[1];
			mat.a[0] = a[3]/det; mat.a[1] = -a[1]/det; mat.a[2] = -a[2]/det; mat.a[3] = a[0]/det;
			set(mat);
		}
	};
	typedef /*__host__ __device__*/ struct mat2 mat2;



	/*__host__ __device__*/ struct mat3 {
		float a[9];
		mat3() {			for (int i = 0; i < 9; i++) a[i] = 0;			}
		mat3(float _a[9]) { for (int i = 0; i < 9; i++) a[i] = _a[i];		}
		mat3(f3 _a, f3 _b, f3 _c) {  //Vectors to COLUMNS of the matrix
			a[0] = _a.x; a[1] = _b.x; a[2] = _c.x;
			a[3] = _a.y; a[4] = _b.y; a[5] = _c.y;
			a[6] = _a.z; a[7] = _b.z; a[8] = _c.z;
		}

		/*__host__ __device__*/ inline f3 operator[](int i) { if (i >= 3) return f3(); return f3(a[0+i], a[3+i], a[6+i]); }

		void set(const mat3 &_a) { for (int i = 0; i < 9; i++) a[i] = _a.a[i]; }
		void cinv() {		for (int i = 0; i < 9; i++) a[i] = 1.0f/a[i];	}
	};
	typedef /*__host__ __device__*/ struct mat3 mat3;



	/*__host__ __device__*/ struct mat4 {
		float a[16];
		mat4() {			for (int i = 0; i < 16; i++) a[i] = 0;			}
		mat4(float _a[16]) { for (int i = 0; i < 16; i++) a[i] = _a[i];		}
		mat4(f4 _a, f4 _b, f4 _c, f4 _d) { //Vectors to COLUMNS of the matrix
			a[0] = _a.x; a[1] = _b.x; a[2] = _c.x; a[3] = _d.x;
			a[4] = _a.y; a[5] = _b.y; a[6] = _c.y; a[7] = _d.y;
			a[8] = _a.z; a[9] = _b.z; a[10] = _c.z; a[11] = _d.z;
			a[12] = _a.w; a[13] = _b.w; a[14] = _c.w; a[15] = _d.w;
		}

		/*__host__ __device__*/ inline f4 operator[](int i) { if (i >= 4) return f4(); return f4(a[0+i], a[4+i], a[8+i], a[12+i]); }

		void set(const mat4 &_a) { for (int i = 0; i < 16; i++) a[i] = _a.a[i]; }
		void cinv() {		for (int i = 0; i < 16; i++) a[i] = 1.0f/a[i];	}
	};
	typedef /*__host__ __device__*/ struct mat4 mat4;
#pragma endregion





typedef /*__host__ __device__*/ struct f3 f3;




//TODO: ---------------------------------------------------------------------------------//
//TODO: --------------------------------- OPERATIONS ------------------------------------//
//TODO: ---------------------------------------------------------------------------------//


#pragma region i2
	/*__host__ __device__*/ inline i2 operator+(const i2 &a, const i2 &b) {		return i2(a.x+b.x, a.y+b.y);	}
	/*__host__ __device__*/ inline i2 operator-(const i2 &a, const i2 &b) {		return i2(a.x-b.x, a.y-b.y);	}
	/*__host__ __device__*/ inline i2 operator*(const i2 &a, const i2 &b) {		return i2(a.x*b.x, a.y*b.y);	}
	/*__host__ __device__*/ inline f2 operator/(const i2 &a, const i2 &b) {		return f2(a.x/b.x, a.y/b.y);	}

	/*__host__ __device__*/ inline i2 operator*(const int &a, const i2 &b) {		return i2(a*b.x, a*b.y);		}
	/*__host__ __device__*/ inline i2 operator*(const i2 &b, const int &a) {		return i2(a*b.x, a*b.y);		}
	/*__host__ __device__*/ inline f2 operator/(const i2 &b, const int &a) {		return f2(b.x/a, b.y/a);		}
	/*__host__ __device__*/ inline f2 operator*(const float &a, const i2 &b) {		return f2(a*b.x, a*b.y);		}
	/*__host__ __device__*/ inline f2 operator*(const i2 &b, const float &a) {		return f2(a*b.x, a*b.y);		}
	/*__host__ __device__*/ inline f2 operator/(const i2 &b, const float &a) {		return f2(b.x/a, b.y/a);		}

#pragma endregion

#pragma region f2
	/*__host__ __device__*/ inline f2 operator+(const f2 &a, const f2 &b) {		return f2(a.x+b.x, a.y+b.y);	}
	/*__host__ __device__*/ inline f2 operator-(const f2 &a, const f2 &b) {		return f2(a.x-b.x, a.y-b.y);	}
	/*__host__ __device__*/ inline f2 operator*(const f2 &a, const f2 &b) {		return f2(a.x*b.x, a.y*b.y);	}
	/*__host__ __device__*/ inline f2 operator/(const f2 &a, const f2 &b) {		return f2(a.x/b.x, a.y/b.y);	}

	/*__host__ __device__*/ inline f2 operator*(const float &a, const f2 &b) {		return f2(a*b.x, a*b.y);		}
	/*__host__ __device__*/ inline f2 operator*(const f2 &b, const float &a) {		return f2(a*b.x, a*b.y);		}
	/*__host__ __device__*/ inline f2 operator/(const f2 &b, const float &a) {		return f2(b.x/a, b.y/a);		}
#pragma endregion

#pragma region i2_f2
	/*__host__ __device__*/ inline f2 operator+(const f2 &a, const i2 &b) {      return f2(a.x+b.x, a.y+b.y);	}
	/*__host__ __device__*/ inline f2 operator-(const f2 &a, const i2 &b) {      return f2(a.x-b.x, a.y-b.y);	}
	/*__host__ __device__*/ inline f2 operator*(const f2 &a, const i2 &b) {      return f2(a.x*b.x, a.y*b.y);	}
	/*__host__ __device__*/ inline f2 operator/(const f2 &a, const i2 &b) {      return f2(a.x/b.x, a.y/b.y);	}
	/*__host__ __device__*/ inline f2 operator+(const i2 &a, const f2 &b) {      return f2(a.x+b.x, a.y+b.y);	}
	/*__host__ __device__*/ inline f2 operator-(const i2 &a, const f2 &b) {      return f2(a.x-b.x, a.y-b.y);	}
	/*__host__ __device__*/ inline f2 operator*(const i2 &a, const f2 &b) {      return f2(a.x*b.x, a.y*b.y);	}
	/*__host__ __device__*/ inline f2 operator/(const i2 &a, const f2 &b) {      return f2(a.x/b.x, a.y/b.y);	}
#pragma endregion





#pragma region i3
	/*__host__ __device__*/ inline i3 operator+(const i3 &a, const i3 &b) {		return i3(a.x+b.x, a.y+b.y, a.z+b.z);	}
	/*__host__ __device__*/ inline i3 operator-(const i3 &a, const i3 &b) {		return i3(a.x-b.x, a.y-b.y, a.z-b.z);	}
	/*__host__ __device__*/ inline i3 operator*(const i3 &a, const i3 &b) {		return i3(a.x*b.x, a.y*b.y, a.z*b.z);	}
	/*__host__ __device__*/ inline f3 operator/(const i3 &a, const i3 &b) {		return f3(a.x/b.x, a.y/b.y, a.z/b.z);	}

	/*__host__ __device__*/ inline i3 operator*(const int &a, const i3 &b) {		return i3(a*b.x, a*b.y, a*b.z);			}
	/*__host__ __device__*/ inline i3 operator*(const i3 &b, const int &a) {		return i3(a*b.x, a*b.y, a*b.z);			}
	/*__host__ __device__*/ inline f3 operator/(const i3 &b, const int &a) {		return f3(b.x/a, b.y/a, b.z/a);			}
	/*__host__ __device__*/ inline f3 operator*(const float &a, const i3 &b) {		return f3(a*b.x, a*b.y, a*b.z);			}
	/*__host__ __device__*/ inline f3 operator*(const i3 &b, const float &a) {		return f3(a*b.x, a*b.y, a*b.z);			}
	/*__host__ __device__*/ inline f3 operator/(const i3 &b, const float &a) {		return f3(b.x/a, b.y/a, b.z/a);			}

#pragma endregion

#pragma region f3
	/*__host__ __device__*/ inline f3 operator+(const f3 &a, const f3 &b) {		return f3(a.x+b.x, a.y+b.y, a.z+b.z);	}
	/*__host__ __device__*/ inline f3 operator-(const f3 &a, const f3 &b) {		return f3(a.x-b.x, a.y-b.y, a.z-b.z);	}
	/*__host__ __device__*/ inline f3 operator*(const f3 &a, const f3 &b) {		return f3(a.x*b.x, a.y*b.y, a.z*b.z);	}
	/*__host__ __device__*/ inline f3 operator/(const f3 &a, const f3 &b) {		return f3(a.x/b.x, a.y/b.y, a.z/b.z);	}

	/*__host__ __device__*/ inline f3 operator*(const float &a, const f3 &b) {		return f3(a*b.x, a*b.y, a*b.z);			}
	/*__host__ __device__*/ inline f3 operator*(const f3 &b, const float &a) {		return f3(a*b.x, a*b.y, a*b.z);			}
	/*__host__ __device__*/ inline f3 operator/(const f3 &b, const float &a) {		return f3(b.x/a, b.y/a, b.z/a);			}
#pragma endregion

#pragma region i3_f3
	/*__host__ __device__*/ inline f3 operator+(const f3 &a, const i3 &b) {      return f3(a.x+b.x, a.y+b.y, a.z+b.z);	}
	/*__host__ __device__*/ inline f3 operator-(const f3 &a, const i3 &b) {      return f3(a.x-b.x, a.y-b.y, a.z-b.z);	}
	/*__host__ __device__*/ inline f3 operator*(const f3 &a, const i3 &b) {      return f3(a.x*b.x, a.y*b.y, a.z*b.z);	}
	/*__host__ __device__*/ inline f3 operator/(const f3 &a, const i3 &b) {      return f3(a.x/b.x, a.y/b.y, a.z/b.z);	}
	/*__host__ __device__*/ inline f3 operator+(const i3 &a, const f3 &b) {      return f3(a.x+b.x, a.y+b.y, a.z+b.z);	}
	/*__host__ __device__*/ inline f3 operator-(const i3 &a, const f3 &b) {      return f3(a.x-b.x, a.y-b.y, a.z-b.z);	}
	/*__host__ __device__*/ inline f3 operator*(const i3 &a, const f3 &b) {      return f3(a.x*b.x, a.y*b.y, a.z*b.z);	}
	/*__host__ __device__*/ inline f3 operator/(const i3 &a, const f3 &b) {      return f3(a.x/b.x, a.y/b.y, a.z/b.z);	}
#pragma endregion





#pragma region i4
	/*__host__ __device__*/ inline i4 operator+(const i4 &a, const i4 &b) {		return i4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);	}
	/*__host__ __device__*/ inline i4 operator-(const i4 &a, const i4 &b) {		return i4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);	}
	/*__host__ __device__*/ inline i4 operator*(const i4 &a, const i4 &b) {		return i4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);	}
	/*__host__ __device__*/ inline f4 operator/(const i4 &a, const i4 &b) {		return f4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);	}

	/*__host__ __device__*/ inline i4 operator*(const int &a, const i4 &b) {		return i4(a*b.x, a*b.y, a*b.z, a*b.w);			}
	/*__host__ __device__*/ inline i4 operator*(const i4 &b, const int &a) {		return i4(a*b.x, a*b.y, a*b.z, a*b.w);			}
	/*__host__ __device__*/ inline f4 operator/(const i4 &b, const int &a) {		return f4(b.x/a, b.y/a, b.z/a, b.w/a);			}
	/*__host__ __device__*/ inline f4 operator*(const float &a, const i4 &b) {		return f4(a*b.x, a*b.y, a*b.z, a*b.w);			}
	/*__host__ __device__*/ inline f4 operator*(const i4 &b, const float &a) {		return f4(a*b.x, a*b.y, a*b.z, a*b.w);			}
	/*__host__ __device__*/ inline f4 operator/(const i4 &b, const float &a) {		return f4(b.x/a, b.y/a, b.z/a, b.w/a);			}

#pragma endregion

#pragma region f4
	/*__host__ __device__*/ inline f4 operator+(const f4 &a, const f4 &b) {		return f4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);     }
	/*__host__ __device__*/ inline f4 operator-(const f4 &a, const f4 &b) {		return f4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);     }
	/*__host__ __device__*/ inline f4 operator*(const f4 &a, const f4 &b) {		return f4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);     }
	/*__host__ __device__*/ inline f4 operator/(const f4 &a, const f4 &b) {		return f4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);     }

	/*__host__ __device__*/ inline f4 operator*(const float &a, const f4 &b) {		return f4(a*b.x, a*b.y, a*b.z, a*b.w);		}
	/*__host__ __device__*/ inline f4 operator*(const f4 &b, const float &a) {		return f4(a*b.x, a*b.y, a*b.z, a*b.w);		}
	/*__host__ __device__*/ inline f4 operator/(const f4 &b, const float &a) {		return f4(b.x/a, b.y/a, b.z/a, b.w/a);		}
#pragma endregion

#pragma region i4_f4
	/*__host__ __device__*/ inline f4 operator+(const f4 &a, const i4 &b) {      return f4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);	}
	/*__host__ __device__*/ inline f4 operator-(const f4 &a, const i4 &b) {      return f4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);	}
	/*__host__ __device__*/ inline f4 operator*(const f4 &a, const i4 &b) {      return f4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);	}
	/*__host__ __device__*/ inline f4 operator/(const f4 &a, const i4 &b) {      return f4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);	}
	/*__host__ __device__*/ inline f4 operator+(const i4 &a, const f4 &b) {      return f4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);	}
	/*__host__ __device__*/ inline f4 operator-(const i4 &a, const f4 &b) {      return f4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);	}
	/*__host__ __device__*/ inline f4 operator*(const i4 &a, const f4 &b) {      return f4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);	}
	/*__host__ __device__*/ inline f4 operator/(const i4 &a, const f4 &b) {      return f4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);	}
#pragma endregion

#pragma region mat2
	/*__host__ __device__*/ inline mat2 operator+(const mat2 &a, const mat2 &b) { mat2 mat; for (int i = 0; i < 4; i++) mat.a[i] = a.a[i] + b.a[i]; return mat; }
	/*__host__ __device__*/ inline mat2 operator-(const mat2 &a, const mat2 &b) { mat2 mat; for (int i = 0; i < 4; i++) mat.a[i] = a.a[i] - b.a[i]; return mat; }
	/*__host__ __device__*/ inline mat2 operator*(const mat2 &a, const mat2 &b) { mat2 mat; for (int i = 0; i < 4; i++) mat.a[i] = a.a[i] * b.a[i]; return mat; }
	/*__host__ __device__*/ inline mat2 operator/(const mat2 &a, const mat2 &b) { mat2 mat; for (int i = 0; i < 4; i++) mat.a[i] = a.a[i] / b.a[i]; return mat; }

	/*__host__ __device__*/ inline mat2 operator*(const float &a, const mat2 &b) { mat2 mat; for (int i = 0; i < 4; i++) mat.a[i] = a * b.a[i]; return mat; }
	/*__host__ __device__*/ inline mat2 operator*(const mat2 &b, const float &a) { mat2 mat; for (int i = 0; i < 4; i++) mat.a[i] = a * b.a[i]; return mat; }
	/*__host__ __device__*/ inline mat2 operator/(const mat2 &a, const float &b) { mat2 mat; for (int i = 0; i < 4; i++) mat.a[i] = a.a[i] / b; return mat; }


#pragma endregion



//TODO: ---------------------------------------------------------------------------------//
//TODO: --------------------------------- FUNCTIONS -------------------------------------//
//TODO: ---------------------------------------------------------------------------------//


#pragma region ints
	/*__host__ __device__*/ inline int dot(const i2 &a, const i2 &b) {		return a.x*b.x+a.y*b.y;					}
	/*__host__ __device__*/ inline float norm(const i2 &a) {				return sqrtf(a.x*a.x+a.y*a.y);			}
	/*__host__ __device__*/ inline int norm2(const i2 &a) {					return a.x*a.x+a.y*a.y;					}
	/*__host__ __device__*/ inline f2 normalized(const i2 &a) {				return a/norm(a);						}
	/*__host__ __device__*/ inline i2 abs(const i2 &a) {					return i2(abs(a.x), abs(a.y));			}
	/*__host__ __device__*/ inline i2 cmin(const i2 &a, const i2 &b) {		return i2(fmin(a.x, b.x), fmin(a.y, b.y));	}
	/*__host__ __device__*/ inline i2 cmax(const i2 &a, const i2 &b) {		return i2(fmax(a.x, b.x), fmax(a.y, b.y));	}
	/*__host__ __device__*/ inline i2 nmin(const i2 &a, const i2 &b) {		return norm2(a) < norm2(b) ? a : b;		}
	/*__host__ __device__*/ inline i2 nmax(const i2 &a, const i2 &b) {		return norm2(a) > norm2(b) ? a : b;		}

	/*__host__ __device__*/ inline int dot(const i3 &a, const i3 &b) {		return a.x*b.x+a.y*b.y+a.z*b.z;					}
	/*__host__ __device__*/ inline float norm(const i3 &a) {				return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);			}
	/*__host__ __device__*/ inline int norm2(const i3 &a) {					return a.x*a.x+a.y*a.y+a.z*a.z;					}
	/*__host__ __device__*/ inline f3 normalize(const i3 &a) {				return a/norm(a);								}
	/*__host__ __device__*/ inline i3 abs(const i3 &a) {					return i3(abs(a.x), abs(a.y), abs(a.z));		}
	/*__host__ __device__*/ inline i3 cmin(const i3 &a, const i3 &b) {		return i3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));	}
	/*__host__ __device__*/ inline i3 cmax(const i3 &a, const i3 &b) {		return i3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));	}
	/*__host__ __device__*/ inline i3 nmin(const i3 &a, const i3 &b) {		return norm2(a) < norm2(b) ? a : b;				}
	/*__host__ __device__*/ inline i3 nmax(const i3 &a, const i3 &b) {		return norm2(a) > norm2(b) ? a : b;				}

#pragma endregion




#pragma region floats
	/*__host__ __device__*/ inline float dot(const f2 &a, const f2 &b) {	return a.x*b.x+a.y*b.y;					}
	/*__host__ __device__*/ inline float norm(const f2 &a) {				return sqrtf(a.x*a.x+a.y*a.y);			}
	/*__host__ __device__*/ inline float norm2(const f2 &a) {				return a.x*a.x+a.y*a.y;					}
	/*__host__ __device__*/ inline f2 normalize(const f2 &a) {				return a/norm(a);						}
	/*__host__ __device__*/ inline f2 abs(const f2 &a) {					return f2(abs(a.x), abs(a.y));		}
	/*__host__ __device__*/ inline f2 cinv(const f2 &a) {					return f2(1.0f/a.x, 1.0f/a.y);	}
	/*__host__ __device__*/ inline f2 cmin(const f2 &a, const f2 &b) {		return f2(fmin(a.x, b.x), fmin(a.y, b.y));	}
	/*__host__ __device__*/ inline f2 cmax(const f2 &a, const f2 &b) {		return f2(fmax(a.x, b.x), fmax(a.y, b.y));	}
	/*__host__ __device__*/ inline f2 nmin(const f2 &a, const f2 &b) {		return norm2(a) < norm2(b) ? a : b;		}
	/*__host__ __device__*/ inline f2 nmax(const f2 &a, const f2 &b) {		return norm2(a) > norm2(b) ? a : b;		}

	/*__host__ __device__*/ inline float dot(const f3 &a, const f3 &b) {	return a.x*b.x+a.y*b.y+a.z*b.z;						}
	/*__host__ __device__*/ inline f3 cross(const f3 &a, const f3 &b) {		return f3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);	}
	/*__host__ __device__*/ inline float norm(const f3 &a) {				return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);				}
	/*__host__ __device__*/ inline float norm2(const f3 &a) {				return a.x*a.x+a.y*a.y+a.z*a.z;						}
	/*__host__ __device__*/ inline f3 normalize(const f3 &a) {				return a/norm(a);									}
	/*__host__ __device__*/ inline f3 cinv(const f3 &a) {					return f3(1.0f/a.x, 1.0f/a.y, 1.0f/a.z);			}
	/*__host__ __device__*/ inline f3 abs(const f3 &a) {					return f3(abs(a.x), abs(a.y), abs(a.z));			}
	/*__host__ __device__*/ inline f3 cmin(const f3 &a, const f3 &b) {		return f3(fmin(a.x, b.x), fmin(b.x, b.y), fmin(a.z, b.z));		}
	/*__host__ __device__*/ inline f3 cmax(const f3 &a, const f3 &b) {		return f3(fmax(a.x, b.x), fmax(b.x, b.y), fmax(a.z, b.z));		}
	/*__host__ __device__*/ inline f3 nmin(const f3 &a, const f3 &b) {		return norm2(a) < norm2(b) ? a : b;					}
	/*__host__ __device__*/ inline f3 nmax(const f3 &a, const f3 &b) {		return norm2(a) > norm2(b) ? a : b;					}

	/*__host__ __device__*/ inline float dot(const f4 &a, const f4 &b) {	return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;					}
	/*__host__ __device__*/ inline float norm(const f4 &a) {				return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);			}
	/*__host__ __device__*/ inline float norm2(const f4 &a) {				return a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w;					}
	/*__host__ __device__*/ inline f4 normalized(const f4 &a) {				return a/norm(a);										}
	/*__host__ __device__*/ inline f4 cinv(const f4 &a) {					return f4(1.0f/a.x, 1.0f/a.y, 1.0f/a.z, 1.0f/a.w);	}
	/*__host__ __device__*/ inline f4 abs(const f4 &a) {					return f4(abs(a.x), abs(a.y), abs(a.z), abs(a.w));	}
	/*__host__ __device__*/ inline f4 cmin(const f4 &a, const f4 &b) {		return f4(fmin(a.x, b.x), fmin(b.x, b.y), fmin(a.z, b.z), fmin(a.w, b.w));	}
	/*__host__ __device__*/ inline f4 cmax(const f4 &a, const f4 &b) {		return f4(fmax(a.x, b.x), fmax(b.x, b.y), fmax(a.z, b.z), fmax(a.w, b.w));	}
	/*__host__ __device__*/ inline f4 nmin(const f4 &a, const f4 &b) {		return norm2(a) < norm2(b) ? a : b;						}
	/*__host__ __device__*/ inline f4 nmax(const f4 &a, const f4 &b) {		return norm2(a) > norm2(b) ? a : b;						}
	/*__host__ __device__*/ inline f4 vec(const f3 &a, const float &b) {	return f4(a.x, a.y, a.z, b);						}
#pragma endregion


#pragma region matrices
	//TODO: element-wise inversion, matrix inversion, matrix multiplication (dot), transposed matrix
	/*__host__ __device__*/ inline float det(const mat2 &a) {							return a.a[0]*a.a[3]-a.a[2]*a.a[1];						}
#pragma endregion




#endif // DEFAULTVEC