"""
Flask web server for K. phaffii codon optimizer.
"""

import sys
import os
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template

from optimizer.optimizer import optimize
from optimizer.restriction import RESTRICTION_ENZYMES

app = Flask(__name__, template_folder='templates')

# Global cancel event — reset at the start of each /optimize call.
# Sufficient for single-user / dev use; not safe for concurrent multi-user.
_cancel_event = threading.Event()


@app.route('/')
def index():
    return render_template('index.html', enzymes=RESTRICTION_ENZYMES)


@app.route('/cancel', methods=['POST'])
def cancel_endpoint():
    _cancel_event.set()
    return jsonify({'success': True})


@app.route('/optimize', methods=['POST'])
def optimize_endpoint():
    global _cancel_event
    _cancel_event = threading.Event()  # fresh event for this request

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body received.'}), 400

        sequence = data.get('sequence', '').strip()
        if not sequence:
            return jsonify({'success': False, 'error': 'No sequence provided.'}), 400

        input_type = data.get('input_type', 'auto')
        selected_enzymes = data.get('selected_enzymes', [])
        user_opts = data.get('options', {})
        options = {
            'avoid_rare_codons': user_opts.get('avoid_rare_codons', True),
        }

        result_box: dict = {'result': None, 'exc': None}

        def _run():
            try:
                result_box['result'] = optimize(
                    raw_input=sequence,
                    input_type=input_type,
                    selected_enzymes=selected_enzymes,
                    options=options,
                    cancel_event=_cancel_event,
                )
            except Exception as exc:
                result_box['exc'] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()  # blocks until optimizer finishes or cancel skips remaining fixers

        if result_box['exc'] is not None:
            exc = result_box['exc']
            if isinstance(exc, ValueError):
                return jsonify({'success': False, 'error': str(exc)}), 400
            return jsonify({'success': False, 'error': f'Internal error: {str(exc)}'}), 500

        result = result_box['result']
        return jsonify({
            'success': True,
            'cancelled': result.get('cancelled', False),
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
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
