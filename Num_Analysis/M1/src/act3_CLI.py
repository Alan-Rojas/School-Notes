import argparse
import numpy as np
import cmath   


# Clase Polynomial:
class Polynomial:
    def __init__(self, coefficients):
        # Vamos a recibir coeficientes en forma de lista y orden descendente. Si un exponente no tiene coeficiente, entonces es igual a 0. 
        # e.g. x^3 + 1 = [3, 0, 0, 1]
        # La IA sugiere que se use el tipo de dato complex para evitar errores. 
        self.coeffs = [complex(c) if isinstance(c, complex) else float(c) for c in coefficients]
        self.degree = len(self.coeffs) - 1 # Por lo mismo de la lista de coeff. 

    def evaluate(self, x):
        """Evalúa el polinomio en x usando el Método de Horner."""
        result = self.coeffs[0]
        for c in self.coeffs[1:]:
            result = result * x + c
        return result

    def derivative(self):
        """Retorna un nuevo Polynomial que es la derivada del polinomio."""
        if self.degree == 0: # Caso base
            return Polynomial([0])
        new_coeffs = [c * (self.degree - i) for i, c in enumerate(self.coeffs[:-1])]
        return Polynomial(new_coeffs)

    def divide_linear(self, root):
        """
        Divide el polinomio entre (x - root) usando división sintética.
        """
        if self.degree < 1:
            return self, 0
            
        q = []
        val = 0
        for c in self.coeffs:
            val = val * root + c
            q.append(val)
            
        remainder = q.pop()
        
        return Polynomial(q), remainder

    def divide_quadratic(self, p, q_val): # Función Helper con IA. 
        """
        Divide el polinomio entre (x^2 + p*x + q_val).
        Útil para deflactar raíces complejas conjugadas r y r*:
        (x - r)(x - r*) = x^2 - 2*Re(r)x + |r|^2
        Por lo tanto: p = -2*Re(r) y q_val = |r|^2
        Retorna (cociente_Polynomial, residuo_lista)
        """
        if self.degree < 2:
            return Polynomial([0]), self.coeffs

        quotient_coeffs = []
        # Trabajamos con una copia de los coeficientes para el residuo
        rem = list(self.coeffs)
        
        for i in range(len(rem) - 2):
            c = rem[i]
            quotient_coeffs.append(c)
            rem[i] = 0
            rem[i+1] -= c * p
            rem[i+2] -= c * q_val
            
        return Polynomial(quotient_coeffs), rem[-2:]

#Funciones como métodos.      
def newton_raphson(poly, x0, tol=1e-3, max_iter=100):
    """
    Encuentra una raíz usando el método de Newton-Raphson.
    f(x_{n+1}) = x_n - f(x_n) / f'(x_n)
    """
    x = x0
    deriv = poly.derivative()
    
    for _ in range(max_iter):
        fx = poly.evaluate(x)
        if abs(fx) < tol:
            return x
        
        dfx = deriv.evaluate(x)
        if dfx == 0:
            raise ValueError("Derivada cero encontrada.")
            
        x_next = x - fx / dfx
        if abs(x_next - x) < tol:
            return x_next
        x = x_next
        
    raise Exception("Newton-Raphson no convergió.")

def muller(poly, x0, x1, x2, tol=1e-3, max_iter=100):
    for _ in range(max_iter):
        f0, f1, f2 = poly.evaluate(x0), poly.evaluate(x1), poly.evaluate(x2)
        
        h1 = x1 - x0
        h2 = x2 - x1
        
        # Evitar divisiones por cero en los deltas si los puntos colapsan
        if h1 == 0 or h2 == 0:
            x2 += 0.01j # Perturbación compleja
            continue
            
        delta1 = (f1 - f0) / h1
        delta2 = (f2 - f1) / h2
        
        a = (delta2 - delta1) / (h2 + h1)
        b = a * h2 + delta2
        c = f2
        
        disc = cmath.sqrt(b**2 - 4*a*c)
        
        if abs(b + disc) > abs(b - disc):
            den = b + disc
        else:
            den = b - disc
            
        
        if abs(den) < 1e-14:
            x2 += 0.01 + 0.01j # Perturbación para salir del loop
            continue
            
        dx = -2 * c / den
        x3 = x2 + dx
        
        if abs(dx) < tol:
            return x3
            
        x0, x1, x2 = x1, x2, x3
        
    raise Exception("Müller no convergió.")

def find_all_roots_newton(poly, tol=1e-3):
    roots = []
    current_poly = poly
    guess_pool = np.linspace(-10, 10, 100).tolist() + [0] 
    
    while current_poly.degree > 0:
        if current_poly.degree == 1:
            r = -current_poly.coeffs[1] / current_poly.coeffs[0]
            roots.append(r)
            break
            
        root_found = False
        for guess in guess_pool:
            try:
                root = newton_raphson(current_poly, guess, tol, max_iter=50)
                # Refinamiento final con el polinomio ORIGINAL para eliminar el error de deflación
                try:
                    root = newton_raphson(poly, root, tol/10, max_iter=10)
                except:
                    pass # Si el refinamiento falla, nos quedamos con la raíz deflactada
                
                roots.append(root)
                current_poly, _ = current_poly.divide_linear(root)
                root_found = True
                break
            except:
                continue
                
        if not root_found:
            print(f"Advertencia: Newton se estancó en grado {current_poly.degree}.")
            break
            
    return roots

def find_all_roots_muller(poly, tol=1e-3):
    """Encuentra TODAS las raíces usando Müller y deflación."""
    roots = []
    current_poly = poly
    
    while current_poly.degree > 0:
        # Si llegamos a grado 1, lo resolvemos directamente: ax + b = 0 -> x = -b/a
        if current_poly.degree == 1:
            r = -current_poly.coeffs[1] / current_poly.coeffs[0]
            roots.append(r)
            break
            
        try:
            # Semillas iniciales por defecto para Müller (-1, 0, 1) suelen ser estables
            root = muller(current_poly, -1, 0, 1, tol)
            
            # Si la raíz es significativamente compleja
            if abs(root.imag) > tol:
                roots.append(root)
                roots.append(root.conjugate())
                # Deflación cuadrática: p = -2*Re(r), q = |r|^2
                p = -2 * root.real
                q_val = (root.real**2 + root.imag**2)
                current_poly, _ = current_poly.divide_quadratic(p, q_val)
            else:
                # Raíz real
                root_real = root.real
                roots.append(root_real)
                current_poly, _ = current_poly.divide_linear(root_real)
        except Exception as e:
            print(f"Müller se detuvo prematuramente: {e}")
            break
            
    return roots


def format_root(root):
    """Función auxiliar para imprimir raíces de forma elegante."""
    if isinstance(root, complex):
        # Ocultar la parte imaginaria si es prácticamente cero
        if abs(root.imag) < 1e-10:
            return f"{root.real:.6f}"
        sign = "+" if root.imag >= 0 else "-"
        return f"{root.real:.6f} {sign} {abs(root.imag):.6f}j"
    return f"{root:.6f}"

def main():
    # 1. Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description="Calculadora Numérica de Raíces Polinomiales",
        epilog="Ejemplo de uso: python main.py 1 -1 2 -2 -m muller"
    )
    
    # Argumento posicional: los coeficientes (nargs='+' permite 1 o más valores)
    parser.add_argument(
        'coeffs', 
        metavar='C', 
        type=float, 
        nargs='+',
        help='Coeficientes del polinomio en orden descendente (ej. 1 0 0 3 para x^3 + 3)'
    )
    
    # Argumento opcional pero requerido: el método numérico
    parser.add_argument(
        '-m', 
        type=str, 
        choices=['newton', 'muller'], 
        required=True,
        help='Método numérico a utilizar ("newton" o "muller")'
    )
    
    # Argumento opcional: la tolerancia
    parser.add_argument(
        '-t', 
        type=float, 
        default=1e-3,
        help='Tolerancia de error para la convergencia (default: 0.001)'
    )

    # 2. Parsear los argumentos ingresados por el usuario
    args = parser.parse_args()

    # 3. Construir el polinomio y ejecutar la lógica
    poly = Polynomial(args.coeffs)
    
    print("\n" + "="*50)
    print(f" Análisis Polinomial ({args.method.capitalize()})")
    print("="*50)
    print(f"Coeficientes ingresados : {args.coeffs}")
    print(f"Grado del polinomio     : {poly.degree}")
    print(f"Tolerancia configurada  : {args.tol}")
    print("-" * 50)

    # 4. Llamar al solver correspondiente
    try:
        if args.method == 'newton':
            roots = find_all_roots_newton(poly, tol=args.tol)
        elif args.method == 'muller':
            roots = find_all_roots_muller(poly, tol=args.tol)
            
        # Imprimir resultados del algoritmo
        print("\n[+] Raíces Computadas:")
        for i, root in enumerate(roots):
            print(f"    x_{i+1} = {format_root(root)}")
            
    except Exception as e:
        print(f"\n[!] Error durante el cálculo: {e}")

    # 5. Verificación de "Ground Truth" con NumPy
    print("\n[-] Verificación con NumPy (Ground Truth):")
    np_roots = np.roots(args.coeffs)
    for i, root in enumerate(np_roots):
         print(f"    Exacta_{i+1} = {format_root(root)}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()