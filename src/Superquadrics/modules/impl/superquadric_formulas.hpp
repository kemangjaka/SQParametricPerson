#pragma once

/// Need this for the ceres::abs
#include <ceres/jet.h>

#include "../superquadric_formulas.h"


////////////////////////////////////////////////////////////////////////////////
template <typename Scalar> inline void
sq::clampParameters (Scalar &e1_clamped, Scalar &e2_clamped)
{
  if (e1_clamped < Scalar (0.1))
    e1_clamped = Scalar (0.1);
  else if (e1_clamped > Scalar (1.9))
    e1_clamped = Scalar (1.9);

  if (e2_clamped < Scalar (0.1))
    e2_clamped = Scalar (0.1);
  else if (e2_clamped > Scalar (1.9))
    e2_clamped = Scalar (1.9);
}


////////////////////////////////////////////////////////////////////////////////
template <typename Scalar> inline Scalar
sq::superquadric_function (const Scalar &x, const Scalar &y, const Scalar &z,
                           const Scalar &e1, const Scalar &e2,
                           const Scalar &a, const Scalar &b, const Scalar &c)
{
  Scalar e1_clamped = e1,
      e2_clamped = e2;
  clampParameters (e1_clamped, e2_clamped);

  Scalar term_1 = pow (ceres::abs (x / a), Scalar (2.) / e2_clamped);
  Scalar term_2 = pow (ceres::abs (y / b), Scalar (2.) / e2_clamped);
  Scalar term_3 = pow (ceres::abs (z / c), Scalar (2.) / e1_clamped);
  Scalar superellipsoid_f = pow (ceres::abs (term_1 + term_2), e2_clamped / e1_clamped) + term_3;

  Scalar value = abs(superellipsoid_f - Scalar (1.));

  return (value);
}


////////////////////////////////////////////////////////////////////////////////
template <typename Scalar> inline Scalar
sq::normalized_distance(const Scalar &x, const Scalar &y, const Scalar &z,
	const Scalar &e1, const Scalar &e2,
	const Scalar &a, const Scalar &b, const Scalar &c)
{
	Scalar e1_clamped = e1,
		e2_clamped = e2;
	clampParameters(e1_clamped, e2_clamped);

	Scalar term_1 = pow(ceres::abs(x / a), Scalar(2.) / e2_clamped);
	Scalar term_2 = pow(ceres::abs(y / b), Scalar(2.) / e2_clamped);
	Scalar term_3 = pow(ceres::abs(z / c), Scalar(2.) / e1_clamped);
	Scalar superellipsoid_f = pow(ceres::abs(term_1 + term_2), e2_clamped / e1_clamped) + term_3;

	Scalar value = pow(superellipsoid_f, -e1_clamped / Scalar(2.));

	return (value);
}

////////////////////////////////////////////////////////////////////////////////
template <typename Scalar> inline Scalar
sq::superquadric_function_maxout(const Scalar &x, const Scalar &y, const Scalar &z,
	const Scalar &e1, const Scalar &e2,
	const Scalar &a, const Scalar &b, const Scalar &c)
{
	Scalar e1_clamped = e1,
		e2_clamped = e2;
	clampParameters(e1_clamped, e2_clamped);

	Scalar term_1 = pow(ceres::abs(x / a), Scalar(2.) / e2_clamped);
	Scalar term_2 = pow(ceres::abs(y / b), Scalar(2.) / e2_clamped);
	Scalar term_3 = pow(ceres::abs(z / c), Scalar(2.) / e1_clamped);
	Scalar superellipsoid_f = pow(ceres::abs(term_1 + term_2), e2_clamped / e1_clamped) + term_3;

	Scalar value = abs(superellipsoid_f - Scalar(1.));
	if (value > Scalar(0.6))
		value = Scalar(0.0);
	return (value);
}


////////////////////////////////////////////////////////////////////////////////
template <typename Scalar> inline Scalar
sq::superquadric_function_scale_weighting (const Scalar &x, const Scalar &y, const Scalar &z,
                                           const Scalar &e1, const Scalar &e2,
                                           const Scalar &a, const Scalar &b, const Scalar &c)
{  
  Scalar e1_clamped = e1,
      e2_clamped = e2;
  clampParameters (e1_clamped, e2_clamped);

  Scalar term_1 = pow (ceres::abs (x / a), Scalar (2.) / e2_clamped);
  Scalar term_2 = pow (ceres::abs (y / b), Scalar (2.) / e2_clamped);
  Scalar term_3 = pow (ceres::abs (z / c), Scalar (2.) / e1_clamped);
  Scalar superellipsoid_f = pow (ceres::abs (term_1 + term_2), e2_clamped / e1_clamped) + term_3;

  Scalar value = (abs(pow (superellipsoid_f, e1_clamped) - Scalar (1.))) * pow (a*b*c, Scalar (0.25));
  //Scalar value = (abs(pow(superellipsoid_f, e1_clamped / Scalar(2.)) - Scalar(1.))) * pow(a*b*c,0.65);
  return (value);
}



////////////////////////////////////////////////////////////////////////////////
template<typename Scalar> inline void
sq::superquadric_derivative (const Scalar &x, const Scalar &y, const Scalar &z,
                             const Scalar &e1, const Scalar &e2,
                             const Scalar &a, const Scalar &b, const Scalar &c,
                             const Scalar &tx, const Scalar &ty, const Scalar &tz,
                             const Scalar &ax, const Scalar &ay, const Scalar &az,
                             Scalar &dS_de1, Scalar &dS_de2,
                             Scalar &dS_da, Scalar &dS_db, Scalar &dS_dc,
                             Scalar &dS_dtx, Scalar &dS_dty, Scalar &dS_dtz,
                             Scalar &dS_dax, Scalar &dS_day, Scalar &dS_daz)
{
  dS_de1 = pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * (0.5000000000e0 * log(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) + 0.5000000000e0 * e1 * (-pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * e2 * pow(e1, -0.2e1) * log(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) - 0.20e1 * pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1) * pow(e1, -0.2e1) * log(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c))) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1))) * pow(a * b * c, 0.25e0);
  dS_de2 = 0.5000000000e0 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * e1 * pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * (0.1e1 / e1 * log(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) + e2 / e1 * (-0.20e1 * pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) * pow(e2, -0.2e1) * log(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a)) - 0.20e1 * pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2) * pow(e2, -0.2e1) * log(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b))) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2))) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0);
  dS_da = -0.1000000000e1 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) * fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) / ((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) * (cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) * pow(a, -0.2e1) / fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0) + 0.25e0 * (pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) - 0.1e1) * pow(a * b * c, -0.75e0) * b * c;
  dS_db = -0.1000000000e1 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2) * fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) * ((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) * pow(b, -0.2e1) / fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0) + 0.25e0 * (pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) - 0.1e1) * pow(a * b * c, -0.75e0) * a * c;
  dS_dc = -0.1000000000e1 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1) * fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / (((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) * ((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) * pow(c, -0.2e1) / fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0) + 0.25e0 * (pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) - 0.1e1) * pow(a * b * c, -0.75e0) * a * b;
  dS_dtx = 0.1000000000e1 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) * fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) / ((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) / a / fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0);
  dS_dty = 0.1000000000e1 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2) * fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / b / fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0);
  dS_dtz = 0.1000000000e1 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1) * fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / (((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / c / fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0);
  dS_dax = 0.5000000000e0 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * e1 * (0.20e1 * pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) / e1 * pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2) * fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) * ((-cos(ax) * sin(ay) * cos(az) - sin(ax) * sin(az)) * x + (cos(ax) * sin(ay) * sin(az) - sin(ax) * cos(az)) * y - cos(ax) * cos(ay) * z) / b / fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) + 0.20e1 * pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1) / e1 * fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / (((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) * ((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z) / c / fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c)) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0);
  dS_day = 0.5000000000e0 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * e1 * (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * e2 / e1 * (0.20e1 * pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) / e2 * fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) / ((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) * (-sin(ay) * cos(az) * x + sin(ay) * sin(az) * y + cos(ay) * z) / a / fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) + 0.20e1 * pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2) / e2 * fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) * (-sin(ax) * cos(ay) * cos(az) * x + sin(ax) * cos(ay) * sin(az) * y + sin(ax) * sin(ay) * z) / b / fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b)) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) + 0.20e1 * pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1) / e1 * fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / (((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) * (cos(ax) * cos(ay) * cos(az) * x - cos(ax) * cos(ay) * sin(az) * y - cos(ax) * sin(ay) * z) / c / fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c)) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0);
  dS_daz = 0.5000000000e0 * pow(pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1), 0.5000000000e0 * e1) * e1 * (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) * e2 / e1 * (0.20e1 * pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) / e2 * fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) / ((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) * (-cos(ay) * sin(az) * x - cos(ay) * cos(az) * y) / a / fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a) + 0.20e1 * pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2) / e2 * fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) / (((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b) * ((sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * x + (sin(ax) * sin(ay) * cos(az) - cos(ax) * sin(az)) * y) / b / fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b)) / (pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2)) + 0.20e1 * pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1) / e1 * fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) / (((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c) * ((-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * x + (-cos(ax) * sin(ay) * cos(az) - sin(ax) * sin(az)) * y) / c / fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c)) / (pow(pow(fabs((cos(ay) * cos(az) * x - cos(ay) * sin(az) * y + sin(ay) * z + tx) / a), 0.20e1 / e2) + pow(fabs(((-sin(ax) * sin(ay) * cos(az) + cos(ax) * sin(az)) * x + (sin(ax) * sin(ay) * sin(az) + cos(ax) * cos(az)) * y - sin(ax) * cos(ay) * z + ty) / b), 0.20e1 / e2), e2 / e1) + pow(fabs(((cos(ax) * sin(ay) * cos(az) + sin(ax) * sin(az)) * x + (-cos(ax) * sin(ay) * sin(az) + sin(ax) * cos(az)) * y + cos(ax) * cos(ay) * z + tz) / c), 0.20e1 / e1)) * pow(a * b * c, 0.25e0);
}


////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename Scalar> Scalar
sq::computeSuperQuadricError (typename pcl::PointCloud<PointT>::ConstPtr cloud,
                              const Scalar &e1, const Scalar &e2,
                              const Scalar &a, const Scalar &b, const Scalar &c,
                              const Eigen::Matrix<Scalar, 4, 4> &transform)
{
  Scalar error = 0.;
  for (size_t i = 0; i < cloud->size (); ++i)
  {
    Eigen::Matrix<Scalar, 4, 1> xyz (static_cast<Scalar> ((*cloud)[i].x),
                                     static_cast<Scalar> ((*cloud)[i].y),
                                     static_cast<Scalar> ((*cloud)[i].z),
                                     static_cast<Scalar> (1.));
    Eigen::Matrix<Scalar, 4, 1> xyz_tr = transform * xyz;
    double op = (Eigen::Matrix<Scalar, 3, 1> (xyz_tr[0], xyz_tr[1], xyz_tr[2])).norm ();

    double val = op * superquadric_function_scale_weighting (xyz_tr[0], xyz_tr[1], xyz_tr[2], e1, e2, a, b, c);
    error += val * val;
  }

  error /= static_cast<Scalar> (cloud->size ());

  return (error);
}


///////////////////////////////////////////////////////////////////////////////////
template<typename Scalar> Scalar
sq::betaFunction (Scalar x, Scalar y)
{
  return (sqrt (2 * M_PI) * pow (x, (x - 0.5)) * pow (y, y - 0.5) / pow (x+y, x+y - 0.5));
}


////////////////////////////////////////////////////////////////////////////////
template<typename Scalar> Scalar
sq::computeSuperQuadricVolume (const Scalar &e1, const Scalar &e2,
                               const Scalar &a, const Scalar &b, const Scalar &c)
{
  Scalar volume = 2. * e1 * e1 * a * b * c
      * betaFunction (fabs (e1/2. + 1), fabs (e1))
      * betaFunction (fabs (e2 / 2.), fabs (e2 / 2.));

  return (volume);
}
