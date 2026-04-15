"""
Flask web server for K. phaffii codon optimizer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template

from optimizer.optimizer import optimize
from optimizer.restriction import RESTRICTION_ENZYMES

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html', enzymes=RESTRICTION_ENZYMES)


@app.route('/optimize', methods=['POST'])
def optimize_endpoint():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body received.'}), 400

        sequence = data.get('sequence', '').strip()
        if not sequence:
            return jsonify({'success': False, 'error': 'No sequence provided.'}), 400

        input_type = data.get('input_type', 'auto')
        selected_enzymes = data.get('selected_enzymes', [])
        options = data.get('options', {
            'avoid_rare_codons': True,
            'check_hairpin': True,
            'check_repeats': True,
        })

        result = optimize(
            raw_input=sequence,
            input_type=input_type,
            selected_enzymes=selected_enzymes,
            options=options,
        )

        return jsonify({
            'success': True,
            'input_summary': result['input_summary'],
            'optimized_sequence': result['optimized_sequence'],
            'cai_original': result.get('cai_original'),
            'cai_optimized': result['cai_optimized'],
            'cai_interpretation': result['cai_interpretation'],
            'checks': result['checks'],
            'restriction_sites_removed': result['restriction_sites_removed'],
            'restriction_sites_failed': result['restriction_sites_failed'],
            'codon_changes': result['codon_changes'],
            'warnings': result['warnings'],
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Internal error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
