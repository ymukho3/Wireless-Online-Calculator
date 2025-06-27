from flask import Flask, render_template, request
import math

import google.generativeai as genai


app = Flask(__name__)

genai.configure(api_key="AIzaSyAzswBWIsY2tIaju8-tj63y9H1ai_nCL5c")
model = genai.GenerativeModel("gemini-1.5-flash")

def interpret_result(full_prompt):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "You've hit the daily or per-minute quota. Please try again later."
        else:
            return f"Error from AI: {error_msg}"
###
@app.route('/ai', methods=['GET', 'POST'])
def custom_prompt():
    if request.method == 'POST':
        user_prompt = request.form.get('user_prompt', '').strip()
        if not user_prompt:
            return render_template('ai_prompt.html', error="Prompt cannot be empty.")

        return render_template('ai_prompt.html', prompt=user_prompt, response=ai_explanation)

    return render_template('ai_prompt.html')

@app.route("/results", methods=["POST"])
def results():
    user_result = request.form["result"]
    calculator = request.form.get("calculator", "Calculator")
    prompt = request.form.get("prompt", "")
    full_prompt = f"{prompt.strip()}\n\n{user_result.strip()}"
    
    ai_explanation = interpret_result(full_prompt)
    return render_template("results.html",
                           calculator=calculator,
                           result=user_result,
                           ai_explanation=ai_explanation)

########################################################
# Functions for Question 1 calculations
def calculate_sampling_frequency(signal_bandwidth):
    return 2 * signal_bandwidth  # Nyquist rate

def calculate_quantization_rate(signal_bandwidth, number_of_quantizer_bits):
    return signal_bandwidth * number_of_quantizer_bits  

def calculate_source_encoder_rate(quantization_rate, source_encoder_compression_rate):
 return quantization_rate * source_encoder_compression_rate
def calculate_channel_encoder_rate(source_encoder_rate, channel_encoder_rate, voice_segment):
 return (source_encoder_rate / channel_encoder_rate) / voice_segment

def calculate_interleaver_rate(channel_encoder_bit_rate):
    return channel_encoder_bit_rate 

def calculate_burst_formating_rate(voice_segment, channel_encoder_bit_rate):
    needed_burst = channel_encoder_bit_rate / 114
    overhead_bits = needed_burst * (3 + 1 + 26 + 1 + 3 + 8.2496)
    total_bits = channel_encoder_bit_rate + overhead_bits
    return total_bits / voice_segment

# Route for the index page 
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# Route for operations page
@app.route('/operations')
def operations():
    return render_template('operations.html')

# ***************************************************QUESTION1**********************************************************
# Route for Question 1 page
@app.route('/question1', methods=['GET', 'POST'])
def question1():
    if request.method == 'POST':
        try:
            signal_bandwidth = float(request.form['signal_bandwidth'])
            signal_bandwidth_unit = request.form['signal_bandwidth_unit']
            # Convert kHz to Hz if necessary
            if signal_bandwidth_unit == 'kHz':
                signal_bandwidth *= 1000

            number_of_quantizer_bits = int(request.form['number_of_quantizer_bits'])
            if number_of_quantizer_bits <= 0:
                raise ValueError("Number of quantizer bits must be a positive integer.")

            source_encoder_compression_rate = float(request.form['source_encoder_compression_rate'])
            if source_encoder_compression_rate <= 0:
                raise ValueError("Source encoder compression rate must be positive.")

            channel_encoder_rate = float(request.form['channel_encoder_rate'])
            if channel_encoder_rate <= 0:
                raise ValueError("Channel encoder rate must be positive.")

 

            voice_segment = int(request.form['voice_segment'])
            if voice_segment <= 0:
                raise ValueError("Voice segment must be a positive integer.")

            # Store in globals for function access
            globals()['signal_bandwidth'] = signal_bandwidth
            globals()['number_of_quantizer_bits'] = number_of_quantizer_bits

            sampling_frequency = calculate_sampling_frequency(signal_bandwidth)
            quantization_rate = calculate_quantization_rate(signal_bandwidth, number_of_quantizer_bits)
            source_encoder_bit_rate = calculate_source_encoder_rate(quantization_rate, source_encoder_compression_rate)
            channel_encoder_bit_rate = calculate_channel_encoder_rate(source_encoder_bit_rate, channel_encoder_rate, voice_segment)
            interleaver_bit_rate = calculate_interleaver_rate(channel_encoder_bit_rate)
            burst_formating_rate = calculate_burst_formating_rate(voice_segment, channel_encoder_bit_rate)

            results = {
                "Sampling Frequency": f"{sampling_frequency} Hz",
                "Number of Quantization Levels": quantization_rate,
                "Bit Rate at the Output of the Source Encoder": f"{source_encoder_bit_rate:.2f} bits/sec",
                "Bit Rate at the Output of the Channel Encoder": f"{channel_encoder_bit_rate:.2f} bits/sec",
                "Bit Rate at the Output of the Interleaver": f"{interleaver_bit_rate:.2f} bits/sec",
                "Bit Rate at the Output of the Burst Formatting": f"{burst_formating_rate:.2f} bits/sec",
            }

             #Construct prompt text for Gemini AI
            prompt_text = "Please analyze and explain these communication system calculation results clearly and shortly:\n"
            for key, value in results.items():
                prompt_text += f"- {key}: {value}\n"

            # Call Gemini AI to get explanation
            explanation = interpret_result(prompt_text)

            # Pass results and explanation to template
            return render_template('Question1.html', results=results, explanation=explanation)
        
        except ValueError as e:
            return render_template('Question1.html', error=f"Invalid input: {e}")

    return render_template('Question1.html')

# ***************************************************QUESTION2**********************************************************
# Functions for Question 2 calculations
def calculate_bits_per_resource_element(modulation_order):
    #  log₂(modulation_order)
    return int(modulation_order).bit_length() - 1

def calculate_bits_per_ofdm_symbol(bits_per_resource_element, num_sub_carriers):
    return bits_per_resource_element * num_sub_carriers


def calculate_bits_per_resource_block(bits_per_ofdm_symbol, num_ofdm_symbols):
    return bits_per_ofdm_symbol * num_ofdm_symbols


def calculate_max_transmission_rate(bits_per_resource_block, num_resource_blocks, duration_of_rb):
    return (num_resource_blocks * bits_per_resource_block) / duration_of_rb


def calculate_spectral_efficiency(max_transmission_rate, total_bandwidth_hz):
    # to avoid division by zero
    if total_bandwidth_hz == 0:
        raise ValueError("Bandwidth cannot be zero!")
    return max_transmission_rate / total_bandwidth_hz

# Route for Question 2 page
@app.route('/question2', methods=['GET', 'POST'])
def question2():
    if request.method == 'POST':
        try:
            bandwidth_rb = float(request.form['bandwidth_rb'])
            if request.form['bandwidth_rb_unit'] == 'Hz':
                bandwidth_rb /= 1000  # Convert Hz to kHz

            sub_carrier_spacing = float(request.form['subcarrier_spacing'])
            if request.form['subcarrier_spacing_unit'] == 'Hz':
                sub_carrier_spacing /= 1000  # Convert Hz to kHz

            num_ofdm_symbols = int(request.form['num_ofdm_symbols'])

            duration_rb = float(request.form['duration_rb'])
            if request.form['duration_rb_unit'] == 'seconds':
                duration_rb *= 1000  # Convert seconds to milliseconds

            modulation_order = int(request.form['modulation_order'])
            num_parallel_rb = int(request.form['num_parallel_rb'])

            if not (modulation_order & (modulation_order - 1) == 0) and modulation_order != 0:
                raise ValueError("Modulation order must be a power of 2.")

        except ValueError as e:
            return render_template('Question2.html', error=f"Invalid input: {e}",
                                   bandwidth_rb=request.form['bandwidth_rb'],
                                   bandwidth_rb_unit=request.form['bandwidth_rb_unit'],
                                   subcarrier_spacing=request.form['subcarrier_spacing'],
                                   subcarrier_spacing_unit=request.form['subcarrier_spacing_unit'],
                                   num_ofdm_symbols=request.form['num_ofdm_symbols'],
                                   duration_rb=request.form['duration_rb'],
                                   duration_rb_unit=request.form['duration_rb_unit'],
                                   modulation_order=request.form['modulation_order'],
                                   num_parallel_rb=request.form['num_parallel_rb'])

        num_sub_carriers = int(bandwidth_rb / sub_carrier_spacing)
        total_bandwidth_hz = bandwidth_rb * num_parallel_rb

        bits_per_resource_element = calculate_bits_per_resource_element(modulation_order)
        bits_per_ofdm_symbol = calculate_bits_per_ofdm_symbol(bits_per_resource_element, num_sub_carriers)
        bits_per_resource_block = calculate_bits_per_resource_block(bits_per_ofdm_symbol, num_ofdm_symbols)
        max_transmission_rate = calculate_max_transmission_rate(bits_per_resource_block, num_parallel_rb, duration_rb)
        spectral_efficiency = calculate_spectral_efficiency(max_transmission_rate, total_bandwidth_hz)

        results = {
            "Bits per Resource Element": f"{bits_per_resource_element}bits",
            "Bits per OFDM Symbol": f"{bits_per_ofdm_symbol} bits",
            "Bits per Resource Block": f"{bits_per_resource_block} bits",
            "Max Transmission Rate for User": f"{max_transmission_rate * 1000} bits/sec",
            "Spectral Efficiency": f"{spectral_efficiency:.2f} bps/Hz"
        }

        prompt_text = "Analyze the following wireless communication parameters clearly and shortly:\n"
        for key, value in results.items():
           prompt_text += f"- {key}: {value}\n"

        explanation = interpret_result(prompt_text)

        return render_template('Question2.html', results=results, explanation=explanation)

    return render_template('Question2.html')


# ***************************************************QUESTION3**********************************************************
EB_N0_MAP = {
    "BPSK/QPSK": {
        1e-1: 0,
        1e-2: 4,
        1e-3: 7,
        1e-4: 8.3,
        1e-5: 9.6,
        1e-6: 10.5,
        1e-7: 11.6,
        1e-8: 12,
    },
    "8-PSK": {
        1e-1: 0,
        1e-2: 6.5,
        1e-3: 10,
        1e-4: 12,
        1e-5: 12.5,
        1e-6: 14,
        1e-7: 14.7,
        1e-8: 15.6,
    },
    "16-PSK": {
        1e-1: 0,
        1e-2: 10.5,
        1e-3: 14.1,
        1e-4: 16,
        1e-5: 17.7,
        1e-6: 18.3,
        1e-7: 19.2,
        1e-8: 20,
    }
}

def convert_to_db(value, unit):
    if unit == "dB":
        return value
    elif unit in ["watt", "pds", "kelvin"]:
        return 10 * math.log10(value)
    else:
        raise ValueError(f"Invalid unit: {unit}")

def db_to_watt(db_value):
    return 10 ** (db_value / 10)

def calculate_required_eb_n0(modulation_type, ber):
    if modulation_type in EB_N0_MAP:
        ber_map = EB_N0_MAP[modulation_type]
        for key in sorted(ber_map.keys(), reverse=True):
            if ber >= key:
                return ber_map[key]
    raise ValueError(f"Unsupported modulation type: {modulation_type} or BER: {ber}")

def calculate_received_power(m_db, n_f_db, t, r, eb_div_n0):
    k_dB = -228.6  # Boltzmann constant in dB (J/K)
    t_dB = convert_to_db(t, "kelvin")  # Convert temperature to dB
    r_dB = 10 * math.log10(r)  # Convert data rate (bps) to dB directly
    return m_db + k_dB + t_dB + n_f_db + r_dB + eb_div_n0

def calculate_transmit_power(p_r_dB, l_p_db, l_f_db, l_o_db, f_margin_db, g_t_db, g_r_db, a_r_db, a_t_db):
    return p_r_dB + l_p_db + l_f_db + l_o_db + f_margin_db - g_t_db - g_r_db - a_r_db - a_t_db

@app.route('/question3', methods=['GET', 'POST'])
def question3():
    if request.method == 'POST':
        try:
            # Retrieve form data
            l_p = float(request.form['L_p'])
            l_p_unit = request.form['L_p_unit']
            g_t = float(request.form['G_t'])
            g_t_unit = request.form['G_t_unit']
            g_r = float(request.form['G_r'])
            g_r_unit = request.form['G_r_unit']
            r = float(request.form['R'])
            r_unit = request.form['R_unit']
            l_o = float(request.form['L_o'])
            l_o_unit = request.form['L_o_unit']
            l_f = float(request.form['L_f'])
            l_f_unit = request.form['L_f_unit']
            f_margin = float(request.form['F_margin'])
            f_margin_unit = request.form['F_margin_unit']
            a_t = float(request.form['A_t'])
            a_t_unit = request.form['A_t_unit']
            a_r = float(request.form['A_r'])
            a_r_unit = request.form['A_r_unit']
            n_f = float(request.form['N_f'])
            n_f_unit = request.form['N_f_unit']
            t = float(request.form['T'])  # Noise temperature in Kelvin
            m = float(request.form['M'])
            m_unit = request.form['M_unit']
            modulation_type = request.form['modulation_type']
            ber = float(request.form['ber'])

            # Validate units
            valid_units = ["dB", "watt"]
            for unit, field in [
                (l_p_unit, "Path Loss"), (g_t_unit, "Transmit Antenna Gain"), (g_r_unit, "Receive Antenna Gain"),
                (l_o_unit, "Feed Line Loss"), (l_f_unit, "Other Losses"), (f_margin_unit, "Fade Margin"),
                (a_t_unit, "Transmitter Amplifier Gain"), (a_r_unit, "Receiver Amplifier Gain"),
                (n_f_unit, "Noise Figure"), (m_unit, "Link Margin")
            ]:
                if unit not in valid_units:
                    raise ValueError(f"Invalid unit for {field}: {unit}. Must be 'dB' or 'watt'.")

            # Convert R based on the selected unit
            if r_unit == 'kbps':
                r = r * 1e3  # Convert kbps to bps
            elif r_unit == 'Mbps':
                r = r * 1e6  # Convert Mbps to bps
            elif r_unit != 'bps':
                raise ValueError(f"Invalid unit for Data Rate: {r_unit}. Must be 'bps', 'kbps', or 'Mbps'.")

            # Convert values to dB
            l_p_db = convert_to_db(l_p, l_p_unit)
            g_t_db = convert_to_db(g_t, g_t_unit)
            g_r_db = convert_to_db(g_r, g_r_unit)
            l_o_db = convert_to_db(l_o, l_o_unit)
            l_f_db = convert_to_db(l_f, l_f_unit)
            f_margin_db = convert_to_db(f_margin, f_margin_unit)
            a_t_db = convert_to_db(a_t, a_t_unit)
            a_r_db = convert_to_db(a_r, a_r_unit)
            n_f_db = convert_to_db(n_f, n_f_unit)
            m_db = convert_to_db(m, m_unit)

            # Calculate Eb/N0
            eb_div_n0 = calculate_required_eb_n0(modulation_type, ber)

            # Calculate received and transmit power
            received_power_dB = calculate_received_power(m_db, n_f_db, t, r, eb_div_n0)
            transmit_power_dB = calculate_transmit_power(received_power_dB, l_p_db, l_f_db, l_o_db, f_margin_db, g_t_db,
                                                       g_r_db, a_r_db, a_t_db)

            received_power_watt = db_to_watt(received_power_dB)
            transmit_power_watt = db_to_watt(transmit_power_dB)

            results = {
                "Received Power (p_r in dB)": f"{received_power_dB:.5f} dB",
                "Received Power (p_r in Watt)": f"{received_power_watt:.5f} W",
                "Transmit Power (p_t in dB)": f"{transmit_power_dB:.5f} dB",
                "Transmit Power (p_t in Watt)": f"{transmit_power_watt:.5f} W"
            }

            # Step 1: Build prompt for AI explanation
            prompt_text = "Please analyze and explain these satellite link budget calculation results clearly and shortly:\n"
            for key, value in results.items():
                prompt_text += f"- {key}: {value}\n"

            # Step 2: Get explanation from Gemini
            explanation = interpret_result(prompt_text)

            return render_template('Question3.html', results=results, explanation=explanation, form_data=request.form)

        except ValueError as e:
            return render_template('Question3.html', error=f"Error: {str(e)}", form_data=request.form)

    return render_template('Question3.html', form_data={})

# ***************************************************QUESTION4**********************************************************
# Erlang B table as provided
erlang_b_table = {
    0.1: [0.001, 0.046, 0.194, 0.439, 0.762, 1.1, 1.6, 2.1, 2.6, 3.1, 3.7, 4.2, 4.8, 5.4, 6.1, 6.7, 7.4, 8.0, 8.7, 9.4,
          10.1, 10.8, 11.5, 12.2, 13.0, 13.7, 14.4, 15.2, 15.9, 16.7, 17.4, 18.2, 19.0, 19.7, 20.5, 21.3, 22.1, 22.9,
          23.7, 24.4, 25.2, 26.0, 26.8, 27.6, 28.4],
    0.2: [0.002, 0.065, 0.249, 0.535, 0.900, 1.3, 1.8, 2.3, 2.9, 3.4, 4.0, 4.6, 5.3, 5.9, 6.6, 7.3, 7.9, 8.6, 9.4, 10.1,
          10.8, 11.5, 12.3, 13.0, 13.8, 14.5, 15.3, 16.1, 16.8, 17.6, 18.4, 19.2, 20.0, 20.8, 21.6, 22.4, 23.2, 24.0,
          24.8, 25.6, 26.4, 27.2, 28.1, 28.9, 29.7],
    0.5: [0.005, 0.105, 0.349, 0.701, 1.132, 1.6, 2.2, 2.7, 3.3, 4.0, 4.6, 5.3, 6.0, 6.7, 7.4, 8.1, 8.8, 9.6, 10.3,
          11.1, 11.9, 12.6, 13.4, 14.2, 15.0, 15.8, 16.6, 17.4, 18.2, 19.0, 19.9, 20.7, 21.5, 22.3, 23.2, 24.0, 24.8,
          25.7, 26.5, 27.4, 28.2, 29.1, 29.9, 30.8, 31.7],
    1.0: [0.010, 0.153, 0.455, 0.869, 1.361, 1.9, 2.5, 3.1, 3.8, 4.5, 5.2, 5.9, 6.6, 7.4, 8.1, 8.9, 9.7, 10.4, 11.2,
          12.0, 12.8, 13.7, 14.5, 15.3, 16.1, 17.0, 17.8, 18.6, 19.5, 20.3, 21.2, 22.0, 22.9, 23.8, 24.6, 25.5, 26.4,
          27.3, 28.1, 29.0, 29.9, 30.8, 31.7, 32.5, 33.4],
    1.2: [0.012, 0.168, 0.489, 0.922, 1.431, 2.0, 2.6, 3.2, 3.9, 4.6, 5.3, 6.1, 6.8, 7.6, 8.3, 9.1, 9.9, 10.7, 11.5,
          12.3, 13.1, 14.0, 14.8, 15.6, 16.5, 17.3, 18.2, 19.0, 19.9, 20.7, 21.6, 22.5, 23.3, 24.2, 25.1, 26.0, 26.8,
          27.7, 28.6, 29.5, 30.4, 31.3, 32.2, 33.1, 34.0],
    1.3: [0.013, 0.176, 0.505, 0.946, 1.464, 2.0, 2.7, 3.3, 4.0, 4.7, 5.4, 6.1, 6.9, 7.7, 8.4, 9.2, 10.0, 10.8, 11.6,
          12.4, 13.3, 14.1, 14.9, 15.8, 16.6, 17.5, 18.3, 19.2, 20.0, 20.9, 21.8, 22.6, 23.5, 24.4, 25.3, 26.2, 27.0,
          27.9, 28.8, 29.7, 30.6, 31.5, 32.4, 33.3, 34.2],
    1.5: [0.020, 0.190, 0.530, 0.990, 1.520, 2.1, 2.7, 3.4, 4.1, 4.8, 5.5, 6.3, 7.0, 7.8, 8.6, 9.4, 10.2, 11.0, 11.8,
          12.6, 13.5, 14.3, 15.2, 16.0, 16.9, 17.7, 18.6, 19.5, 20.3, 21.2, 22.1, 22.9, 23.8, 24.7, 25.6, 26.5, 27.4,
          28.3, 29.2, 30.1, 31.0, 31.9, 32.8, 33.7, 34.6],
    2.0: [0.020, 0.223, 0.602, 1.092, 1.657, 2.3, 2.9, 3.6, 4.3, 5.1, 5.8, 6.6, 7.4, 8.2, 9.0, 9.8, 10.7, 11.5, 12.3,
          13.2, 14.0, 14.9, 15.8, 16.6, 17.5, 18.4, 19.3, 20.2, 21.0, 21.9, 22.8, 23.7, 24.6, 25.5, 26.4, 27.3, 28.3,
          29.2, 30.1, 31.0, 31.9, 32.8, 33.8, 34.7, 35.6],
    3.0: [0.031, 0.282, 0.715, 1.259, 1.875, 2.5, 3.2, 4.0, 4.7, 5.5, 6.3, 7.1, 8.0, 8.8, 9.6, 10.5, 11.4, 12.2, 13.1,
          14.0, 14.9, 15.8, 16.7, 17.6, 18.5, 19.4, 20.3, 21.2, 22.1, 23.1, 24.0, 24.9, 25.8, 26.8, 27.7, 28.6, 29.6,
          30.5, 31.5, 32.4, 33.4, 34.3, 35.3, 36.2, 37.2],
    5.0: [0.053, 0.381, 0.899, 1.525, 2.218, 3.0, 3.7, 4.5, 5.4, 6.2, 7.1, 8.0, 8.8, 9.7, 10.6, 11.5, 12.5, 13.4, 14.3,
          15.2, 16.2, 17.1, 18.1, 19.0, 20.0, 20.9, 21.9, 22.9, 23.8, 24.8, 25.8, 26.7, 27.7, 28.7, 29.7, 30.7, 31.6,
          32.6, 33.6, 34.6, 35.6, 36.6, 37.6, 38.6, 39.6],
    7.0: [0.075, 0.470, 1.057, 1.784, 2.504, 3.3, 4.1, 5.0, 5.9, 6.8, 7.7, 8.6, 9.5, 10.5, 11.4, 12.4, 13.4, 14.3, 15.3,
          16.3, 17.3, 18.2, 19.2, 20.2, 21.2, 22.2, 23.2, 24.2, 25.2, 26.2, 27.2, 28.2, 29.3, 30.3, 31.3, 32.3, 33.3,
          34.4, 35.4, 36.4, 37.4, 38.4, 39.5, 40.5, 41.5],
    10.0: [0.111, 0.595, 1.271, 2.045, 2.881, 3.8, 4.7, 5.6, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
           16.6, 17.6, 18.7, 19.7, 20.7, 21.8, 22.8, 23.9, 24.9, 26.0, 27.1, 28.1, 29.2, 30.2, 31.3, 32.4, 33.4, 34.5,
           35.6, 36.6, 37.7, 38.8, 39.9, 40.9, 42.0, 43.1, 44.2],
    15.0: [0.176, 0.796, 1.602, 2.501, 3.454, 4.4, 5.5, 6.5, 7.6, 8.6, 9.7, 10.8, 11.9, 13.0, 14.1, 15.2, 16.3, 17.4,
           18.5, 19.6, 20.8, 21.9, 23.0, 24.2, 25.3, 26.4, 27.6, 28.7, 29.9, 31.0, 32.1, 33.3, 34.4, 35.6, 36.7, 37.9,
           39.0, 40.2, 41.3, 42.5, 43.6, 44.8, 45.9, 47.1, 48.2],
    20.0: [0.250, 1.0, 1.930, 2.95, 4.010, 5.1, 6.2, 7.4, 8.5, 9.7, 10.9, 12.0, 13.2, 14.4, 15.6, 16.8, 18.0, 19.2,
           20.4, 21.6, 22.8, 24.1, 25.3, 26.5, 27.7, 28.9, 30.2, 31.4, 32.6, 33.8, 35.1, 36.3, 37.5, 38.8, 40.0, 41.2,
           42.4, 43.7, 44.9, 46.1, 47.4, 48.6, 49.9, 51.1, 52.3],
    30.0: [0.429, 1.45, 2.633, 3.89, 5.189, 6.5, 7.9, 9.2, 10.6, 12.0, 13.3, 14.7, 16.1, 17.5, 18.9, 20.3, 21.7, 23.1,
           24.5, 25.9, 27.3, 28.7, 30.1, 31.6, 33.0, 34.4, 35.8, 37.2, 38.6, 40.0, 41.5, 42.9, 44.3, 45.7, 47.1, 48.6,
           50.0, 51.4, 52.8, 54.2, 55.7, 57.1, 58.5, 59.9, 61.3]
}

# Helper functions (unchanged)
def convert_to_watt(value, unit):
    if unit == "dB":
        return 10 ** (value / 10)
    elif unit == "watt":
        return value
    elif unit == "microwatt":
        return value * 1e-6
    else:
        raise ValueError("Invalid unit")

def convert_to_meters(value, unit):
    if unit == "km":
        return value * 1000
    elif unit == "m":
        return value
    else:
        raise ValueError("Invalid unit")

def convert_area_to_m2(value, unit):
    if unit == "km2":
        return value * 1e6
    elif unit == "m2":
        return value
    else:
        raise ValueError("Invalid unit")

def convert_calls_per_time_to_minutes(calls, unit):
    if unit == "day":
        return calls / (24 * 60)
    elif unit == "hour":
        return calls / 60
    elif unit == "minute":
        return calls
    elif unit == "second":
        return calls * 60
    else:
        raise ValueError("Invalid unit")

def convert_call_duration_to_minutes(duration, unit):
    if unit == "minute":
        return duration
    elif unit == "hour":
        return duration * 60
    elif unit == "second":
        return duration / 60
    elif unit == "day":
        return duration * 24 * 60
    else:
        raise ValueError("Invalid unit")

def calculate_max_distance(reference_power, reference_distance, path_loss_exp, receiver_sensitivity):
    return reference_distance / ((receiver_sensitivity / reference_power) ** (1 / path_loss_exp))

def calculate_cell_size(max_distance):
    return (3 * math.sqrt(3) * (max_distance ** 2)) / 2

def calculate_number_of_cells(area, cell_size):
    return math.ceil(area / cell_size)

def calculate_traffic_load(subscribers, calls_per_minute, call_duration_minutes):
    au = calls_per_minute * call_duration_minutes
    return subscribers * au

def calculate_traffic_load_per_cell(total_traffic_load, num_of_cells):
    return total_traffic_load / num_of_cells

def calculate_cluster_size_n(path_loss_exp, sir):
    possible_n_values = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]
    threshold = (1 / 3) * ((6 * sir) ** (2 / path_loss_exp))
    for n in possible_n_values:
        if n >= threshold:
            return n
    raise ValueError("No suitable number of cells for each cluster found!")

# Updated calculate_min_carriers_per_cell with validation
def calculate_min_carriers_per_cell(gos, traffic_load_per_cell, timeslots_per_carrier):
    if gos not in erlang_b_table:
        raise ValueError(f"Unsupported GoS value: {gos}%. Supported values: {', '.join(map(str, sorted(erlang_b_table.keys())))}%")
    gos_column = erlang_b_table[gos]
    for channels, traffic in enumerate(gos_column, start=1):
        if traffic > traffic_load_per_cell:
            return math.ceil(channels / timeslots_per_carrier)
    raise ValueError("No suitable number of carriers found!")

# Updated question4 route with GoS conversion
@app.route('/question4', methods=['GET', 'POST'])
def question4():
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            area_unit = request.form['area_unit']
            area_m2 = convert_area_to_m2(area, area_unit)

            subscribers = int(request.form['subscribers'])
            calls_per_time = float(request.form['calls_per_day'])
            calls_per_time_unit = request.form['calls_per_day_unit']
            calls_per_minute = convert_calls_per_time_to_minutes(calls_per_time, calls_per_time_unit)

            call_duration = float(request.form['call_duration'])
            call_duration_unit = request.form['call_duration_unit']
            call_duration_minutes = convert_call_duration_to_minutes(call_duration, call_duration_unit)

            sir = float(request.form['sir'])
            sir_unit = request.form['sir_unit']
            sir_watt = convert_to_watt(sir, sir_unit)

            power_reference = float(request.form['power_reference'])
            power_reference_unit = request.form['power_reference_unit']
            power_reference_watt = convert_to_watt(power_reference, power_reference_unit)

            distance_reference = float(request.form['distance_reference'])
            distance_reference_unit = request.form['distance_reference_unit']
            distance_reference_m = convert_to_meters(distance_reference, distance_reference_unit)

            path_loss_exponent = float(request.form['path_loss_exponent'])
            receiver_sensitivity = float(request.form['receiver_sensitivity'])
            receiver_sensitivity_unit = request.form['receiver_sensitivity_unit']
            receiver_sensitivity_watt = convert_to_watt(receiver_sensitivity, receiver_sensitivity_unit)

            # Convert GoS from probability (e.g., 0.02) to percentage (e.g., 2.0)
            gos = float(request.form['gos']) * 100
            timeslots_per_carrier = int(request.form['timeslots_per_carrier'])

            # Calculate maximum distance
            max_distance = calculate_max_distance(power_reference_watt, distance_reference_m, path_loss_exponent,
                                                  receiver_sensitivity_watt)

            # Calculate maximum cell size
            cell_size = calculate_cell_size(max_distance)

            # Calculate number of cells in the service area
            num_of_cells = calculate_number_of_cells(area_m2, cell_size)

            # Calculate traffic load in the whole cellular system in Erlang
            total_traffic_load = calculate_traffic_load(subscribers, calls_per_minute, call_duration_minutes)

            # Calculate traffic load in each cell in Erlang
            traffic_load_per_cell = calculate_traffic_load_per_cell(total_traffic_load, num_of_cells)

            # Calculate the number of cells in each cluster
            cluster_size_n = calculate_cluster_size_n(path_loss_exponent, sir_watt)

            # Calculate minimum number of carriers needed
            min_carriers = calculate_min_carriers_per_cell(gos, traffic_load_per_cell, timeslots_per_carrier)
            total_min_carriers = cluster_size_n * min_carriers

            results = {
                "Maximum Distance (meters)": f"{max_distance:.3f}",
                "Maximum Cell Size (m²)": f"{cell_size:.2f}",
                "Number of Cells": f"{num_of_cells}",
                "Total Traffic Load (Erlang)": f"{total_traffic_load:.2f}",
                "Traffic Load per Cell (Erlang)": f"{traffic_load_per_cell:.2f}",
                "Number of Cells in Each Cluster": f"{cluster_size_n}",
                "Minimum Number of Carriers Needed": f"{total_min_carriers}"
            }

            # Step 1: Prepare prompt for Gemini AI
            prompt_text = "Please explain and analyze the following cellular network planning results clearly and shortly:\n"
            for key, value in results.items():
                prompt_text += f"- {key}: {value}\n"

            # Step 2: Get AI explanation
            explanation = interpret_result(prompt_text)

            return render_template('Question4.html', results=results, explanation=explanation)

        except ValueError as e:
            return render_template('Question4.html', error=f"Invalid input: {str(e)}", **request.form)

    return render_template('Question4.html')




if __name__ == "__main__":
    app.run(debug=True)
