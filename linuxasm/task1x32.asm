;
; ./x32compile.sh task1x32.asm
; ./x32link.sh task1x32.o
; ./build/task1x32.out
; get result call: echo $?


%include "sys_codes.inc"


global _start               ;must be declared for linker (gcc)

section .data
    val1        dd 1
    val2        dd 3
    val3        dd 5
    outresult   dd 0xFFFFFFFF

; Code goes in text section
section .text

_start:
    mov eax, [val1]
    add eax, [val2]
    add eax, [val3]
    mov [outresult], eax

    ; Terminate program
    mov eax, sys_exit       ; system call number (sys_exit: 1)
    mov ebx, [outresult]    ; program result
    int 80h                 ; call kernel


