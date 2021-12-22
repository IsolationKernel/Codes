#ifndef TRACE_H
#define TRACE_H

/**
  * The trace code is from the following site:
  *
  * https://github.com/GPMueller/trace
  *
  */

#include "exception.h"

// Initiation point for a backtrace
// This is as deep as the trace will go
#define ml_tr_initiate(message) trace::inner::throw_exception(message, __FILE__, __LINE__);

// Propagation point for a backtrace
// The trace will include information on the function in which this is placed
#define ml_tr_propagate(message) trace::inner::rethrow_exception(message, __FILE__, __LINE__);

// End point for a backtrace
// This is where the trace info is accumulated and saved,
// it can then be extracted with trace::latest_message
#define ml_tr_handle(ex) trace::inner::handle_exception(ex, __func__);

#endif // TRACE_H
