#include <windows.h>
#include <delayimp.h>

// /DELAYLOAD:NODE.EXE
#pragma comment(lib, "node.lib")

FARPROC WINAPI delayHook(unsigned dliNotify, PDelayLoadInfo pdli) {
  if (dliNotify == dliNotePreLoadLibrary) {
    if (lstrcmpiA(pdli->szDll, "NODE.EXE") == 0) {
      return (FARPROC)GetModuleHandleA("NODE.EXE");
    }
  }
  return NULL;
}

EXTERN_C PfnDliHook __pfnDliNotifyHook2 = delayHook;
