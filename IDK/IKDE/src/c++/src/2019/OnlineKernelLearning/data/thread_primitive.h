/*************************************************************************
  > File Name: thread.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Sun 22 Sep 2013 03:22:34 PM SGT
  > Functions: Primitives for thread
 ************************************************************************/
#pragma once

using namespace std;

namespace SOL
{
#ifdef _WIN32
#include <Windows.h>
    typedef CRITICAL_SECTION MUTEX;
    typedef CONDITION_VARIABLE CV;
#else
    typedef pthread_mutex_t MUTEX;
    typedef pthread_cond_t CV;
#endif

    void initialize_mutex(MUTEX *pm)
    {
#ifdef _WIN32
        ::InitializeCriticalSection(pm);
#else
        pthread_mutex_init(pm,NULL);
#endif
    }

    void delete_mutex(MUTEX *pm)
    {
#ifdef _WIN32 
        ::DeleteCriticalSection(pm);
#else
        //no operation needed here
#endif
    }

    void initialize_condition_variable(CV *pcv)
    {
#ifdef _WIN32
        ::InitializeConditionVariable(pcv);
#else
        pthread_cond_init(pcv,NULL);
#endif
    }

    void mutex_lock(MUTEX *pm)
    {
		//cout<<"obtain lock"<<endl;
#ifdef _WIN32
        ::EnterCriticalSection(pm);
#else
        pthread_mutex_lock(pm);
#endif
    }
    void mutex_unlock(MUTEX *pm)
    {
		//cout<<"release lock"<<endl;
#ifdef _WIN32
        ::LeaveCriticalSection(pm);
#else
        pthread_mutex_unlock(pm);
#endif
    }

    void condition_variable_wait(CV* pcv, MUTEX *pm)
    {
#ifdef _WIN32
        ::SleepConditionVariableCS(pcv,pm,INFINITE);
#else
        pthread_cond_wait(pcv,pm);
#endif
    }
    void condition_variable_signal(CV *pcv)
    {
#ifdef _WIN32
        ::WakeConditionVariable(pcv);
#else
        pthread_cond_signal(pcv);
#endif
    }
    void condition_variable_signal_all(CV *pcv)
    {
#ifdef _WIN32
        ::WakeAllConditionVariable(pcv);
#else
        pthread_cond_broadcast(pcv);
#endif
    }
    /*
#ifdef _WIN32
    void WaitThread(HANDLE &thread){
        WaitForSingleObject(thread,INFINITE);
    }
#else
    void WaitThread(pthread_t &thread){
        pthread_join(thread, NULL);
    }
#endif
*/
}
