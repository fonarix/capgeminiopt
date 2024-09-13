; find maximum value in array
;
; ./x32compile.sh task4x32.asm
; ./x32link.sh task4x32.o
; ./build/task4x32.out
; echo $?

%include "sys_codes.inc"

global _start                   ;must be declared for linker (gcc)

section .data
    array_uint  dd      111, 121, 37, 31, 18, 74, 110, 12, 13, 12, 17, 2, 5
    array_count equ     ($ - array_uint) / 4

    outresult   dd      0xFFFFFFFF

; Code goes in text section
section .text

; Function
; Parameters: ECX, ESI
; Returns: max in EAX
findmax:
    mov     eax, dword [esi]    ; save first element
    add     esi, 4              ; increment to element size
    dec     ecx

    searchmax:
    mov     ebx, dword [esi]    ; store to register element value
    cmp     eax, ebx
    jge     next_elem
    mov     eax, ebx
    next_elem:
    add     esi, 4
    loop searchmax
    ret


; Function
; Parameters: ECX, ESI
; Returns: min in EAX
findmin:
    mov     eax, dword [esi]    ; save first element
    add     esi, 4              ; increment to element size
    dec     ecx

    searchmin:
    mov     ebx, dword [esi]    ; store to register element value
    cmp     eax, ebx
    jl      next_min_elem
    mov     eax, ebx
    next_min_elem:
    add     esi, 4
    loop searchmin
    ret

_start:
    mov     esi, array_uint
    mov     ecx, array_count
    call    findmax
    mov     [outresult], eax

    ; Terminate program
    mov     ebx, [outresult]    ; program result
    mov     eax, sys_exit       ; system call number (sys_exit: 1)
    int     80h                 ; call kernel

