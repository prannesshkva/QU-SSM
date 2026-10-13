import unittest
import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modeling_qu_ssm import QUSSMConfig, QUSSMBlock

class TestQUSSMParity(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = QUSSMConfig(
            d_model=128,
            d_state=8,
            n_layers=1,
            chunk_size=32,
            num_experts=4,
            moe_top_k=2
        )
        self.block = QUSSMBlock(self.config).to(self.device).eval()

    def test_associative_scan_vs_recurrent_step(self):
        B, L, D = 2, 64, self.config.d_model
        x = torch.randn(B, L, D, device=self.device)

        with torch.no_grad():
            # Parallel scan
            scan_out = self.block(x)[0]

            # Recurrent unroll
            state = None
            step_outputs = []
            for t in range(L):
                out_t, state = self.block.step(x[:, t, :], state)
                step_outputs.append(out_t)
            recurrent_out = torch.stack(step_outputs, dim=1)

            cos_sim = F.cosine_similarity(
                scan_out.reshape(-1, D),
                recurrent_out.reshape(-1, D),
                dim=-1
            ).mean().item()

            self.assertGreaterEqual(cos_sim, 0.99999, f"Cosine similarity {cos_sim} < 0.99999")

if __name__ == "__main__":
    unittest.main()
