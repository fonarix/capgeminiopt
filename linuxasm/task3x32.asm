; find max of 3 numbers
;
; ./x32compile.sh task1x32.asm
; ./x32link.sh task1x32.o
; ./build/task1x32.out
; echo $?

; https://cratecode.com/info/x86-assembly-nasm-stack


%include "sys_codes.inc"


global _start                   ;must be declared for linker (gcc)

section .data
    val1        dd 2
    val2        dd 13
    val3        dd 8
    outresult   dd 0xFFFFFFFF

; Code goes in text section
section .text

; Function
; Parameters: EAX, EBX
; returns sorted values from max to min in EAX, EBX
sort_descending2:
    cmp     eax, ebx
    jge     foundmaxof2
    xchg    eax, ebx
    foundmaxof2:
    ret

; Function
; Parameters: EAX, EBX, ECX
; Returns: max in EAX
maxof3:
    call    sort_descending2
    mov     ebx, ecx
    call    sort_descending2
    ret

; Function
; inputs: EAX, EBX, ECX
; returns: min in EAX
minof3:
    call    sort_descending2
    mov     eax, ecx
    call    sort_descending2
    mov     eax, ebx
    ret

_start:
    mov     eax, [val1]
    mov     ebx, [val2]
    mov     ecx, [val3]

    ;call    maxof3
    call    minof3

    ; Terminate program
    mov     ebx, eax            ; program result
    mov     eax, sys_exit       ; system call number (sys_exit: 1)
    int     80h                 ; call kernel

