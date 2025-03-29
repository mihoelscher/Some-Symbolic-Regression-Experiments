import torch
import sympy
import time
from RationalSR import RationalFunction

def benchmark_model(model_class, functions, x_range=(1, 5), num_epochs=4000, device='cpu', reg = None):
    results = []
    torch.manual_seed(26)
    x_train = torch.linspace(x_range[0], x_range[1], 10000).to(device)

    for idx, function_string in enumerate(functions):
        target_function = sympy.lambdify('x', sympy.sympify(function_string))
        y_train = target_function(x_train)

        # Initialize model
        model = model_class(2, 1).to(device)

        # Train model
        start_time = time.time()
        model.fit(x_train, y_train, num_epochs=num_epochs, regularization_parameter=0.1, verbose=0, regularization_order=reg)
        end_time = time.time()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            scale_factor = 1/model.coeffs_q[0].item()
            model.coeffs_q.data *= scale_factor
            model.coeffs_p.data *= scale_factor
            recovered_function = model.get_function()
            lambda_function = sympy.lambdify('x', recovered_function)
            final_loss = model.losses[-1]

        results.append({
            'target_function': function_string,
            'recovered_function': str(recovered_function),
            'final_loss': final_loss,
            'time_taken': end_time - start_time
        })

        print(f"Test {idx + 1}: Loss = {final_loss:.6f}, Time = {end_time - start_time:.2f}, Recovered function = {recovered_function}")

    return results

if __name__ == '__main__':
    # Define multiple test functions with max degrees 2 and 1
    test_functions = [
        '(2*x**2 + x + 1)/(x + 1)',
        '(1.5*x**2 - 0.5*x + 2)/(x + 0.5)',
        '(x**2 + 2*x + 1)/(x + 2)',
        '(3*x**2 - x + 4)/(x + 3)',
        '(4*x**2 + 3*x - 1)/(x + 4)',
        '(2.7*x**2 - 1.2*x + 3)/(x + 1.5)',
        '(x**2 - x + 5)/(x + 2.5)',
        '(5*x**2 + 2*x - 3)/(x + 3.5)',
        '(3.2*x**2 + 0.8*x + 4.5)/(x + 0.8)',
        '(x**2 + 4*x + 2)/(x + 5)'
    ]

    # Run benchmark
    #results = benchmark_model(RationalFunction, test_functions, reg = None)
    """
    Test 1: Loss = 0.000002, Time = 3.44, Recovered function = (2.0149*x**2 + 1.5215*x + 1.1117)/(1.0*x + 1.3299)
    Test 2: Loss = 0.000104, Time = 2.97, Recovered function = (1.6036*x**2 + 0.2455*x + 3.2506)/(1.0*x + 1.6109)
    Test 3: Loss = 0.000005, Time = 3.03, Recovered function = (0.9852*x**2 + 1.3561*x + 0.6282)/(1.0*x + 1.235)
    Test 4: Loss = 0.000038, Time = 2.99, Recovered function = (3.1909*x**2 + 4.309)/(1.0*x + 4.0797)
    Test 5: Loss = 0.000289, Time = 2.90, Recovered function = (4.0206*x**2 + 1.8356*x)/(1.0*x + 3.6475)
    Test 6: Loss = 0.000700, Time = 2.95, Recovered function = (2.9532*x**2 + 2.2547*x + 2.998)/(1.0*x + 3.8823)
    Test 7: Loss = 0.000018, Time = 2.96, Recovered function = (0.9194*x**2 - 0.9189*x + 3.9034)/(1.0*x + 1.7044)
    Test 8: Loss = 0.000021, Time = 3.00, Recovered function = (5.1666*x**2 + 3.951*x - 4.4973)/(1.0*x + 4.2945)
    Test 9: Loss = 0.000727, Time = 2.99, Recovered function = (3.7243*x**2 + 7.1074*x + 13.5865)/(1.0*x + 4.2935)
    Test 10: Loss = 0.000012, Time = 2.88, Recovered function = (0.9296*x**2 + 0.4448*x + 0.4093)/(1.0*x + 0.5137)
    """
    #benchmark_model(RationalFunction, test_functions, reg=1)
    """
    Test 1: Loss = 0.001222, Time = 3.88, Recovered function = (1.9035*x**2 + 0.799*x + 0.2583)/(1.0*x + 0.5783)
    Test 2: Loss = 0.005178, Time = 3.22, Recovered function = 1.0*(1.3066*x**2 + 0.4572)/x
    Test 3: Loss = 0.000470, Time = 3.27, Recovered function = 1.0*(0.9707*x**2 + 0.2934*x)/x
    Test 4: Loss = 0.012968, Time = 3.27, Recovered function = (2.4126*x**2 + 0.5377*x + 0.4792)/(1.0*x + 1.9878)
    Test 5: Loss = 0.000804, Time = 3.28, Recovered function = 3.5847*x**2/(1.0*x + 2.1082)
    Test 6: Loss = 0.011997, Time = 3.22, Recovered function = (2.2973*x**2 + 0.5494)/(1.0*x + 0.9476)
    Test 7: Loss = 0.010999, Time = 3.35, Recovered function = (0.592*x**2 + 0.5303*x + 0.6263)/(1.0*x + 0.6158)
    Test 8: Loss = 0.024776, Time = 3.31, Recovered function = (4.809*x**2 + 1.0458*x - 1.8588)/(1.0*x + 2.9952)
    Test 9: Loss = 0.002567, Time = 3.34, Recovered function = 1.0*(2.9666*x**2 + 1.6591)/x
    Test 10: Loss = 0.000130, Time = 3.25, Recovered function = 0.9126*x
    """
    benchmark_model(RationalFunction, test_functions, reg=2)
    """
    Test 1: Loss = 0.000346, Time = 4.08, Recovered function = (1.9294*x**2 + 0.4301*x + 0.5069)/(1.0*x + 0.4716)
    Test 2: Loss = 0.004161, Time = 3.50, Recovered function = (1.3477*x**2 + 0.5988*x + 1.0473)/(1.0*x + 0.6873)
    Test 3: Loss = 0.000018, Time = 3.47, Recovered function = (0.9746*x**2 + 0.4816*x + 0.2009)/(1.0*x + 0.2512)
    Test 4: Loss = 0.007981, Time = 3.51, Recovered function = (2.528*x**2 + 0.6827*x + 1.0289)/(1.0*x + 2.4289)
    Test 5: Loss = 0.000243, Time = 3.49, Recovered function = (3.6895*x**2 + 0.4434*x)/(1.0*x + 2.4791)
    Test 6: Loss = 0.004936, Time = 3.41, Recovered function = (2.5109*x**2 + 0.7211*x + 1.2498)/(1.0*x + 1.8805)
    Test 7: Loss = 0.008482, Time = 3.51, Recovered function = (0.582*x**2 + 0.5315*x + 0.7593)/(1.0*x + 0.6287)
    Test 8: Loss = 0.000055, Time = 3.47, Recovered function = (4.8495*x**2 - 1.4765)/(1.0*x + 2.7451)
    Test 9: Loss = 0.001241, Time = 3.47, Recovered function = 1.0*(3.0085*x**2 + 2.1005)/x
    Test 10: Loss = 0.000047, Time = 3.50, Recovered function = 0.9109*x
    """
