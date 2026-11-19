"""Pilot counts of visual media and external links: host rule of the 9 Oct test build vs the render rule (rules.py)."""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rules

P2 = os.path.dirname(os.path.dirname(HERE))
L = pd.read_parquet(os.path.join(P2, 'link_domain_output', 'links_pre_accusation_v1.parquet'))
L['renders'] = rules.link_renders(L)
L['place'] = L['kind'] + ' ' + L['where']
old_visual = L['link_type'].eq('visual')
print('links', len(L))
print(pd.crosstab([L['place'], L['link_type']], L['renders']).to_string())
moved_out = old_visual & ~L['renders']
moved_in = ~old_visual & L['renders']
print(f"\nvisual under the host rule: {old_visual.sum()} links, {L[old_visual].outer_key.nunique()} accounts")
print(f"visual under the render rule: {L['renders'].sum()} links, {L[L['renders']].outer_key.nunique()} accounts")
print(f"no longer visual (now URL links): {moved_out.sum()} links, {L[moved_out].outer_key.nunique()} accounts;",
      L[moved_out].groupby('place').size().to_dict(), L[moved_out].domain.value_counts().head(8).to_dict())
print('  top accounts:', L[moved_out].outer_key.value_counts().head(5).to_dict())
print(f"newly visual: {moved_in.sum()} links, {L[moved_in].outer_key.nunique()} accounts;", L[moved_in].link_type.value_counts().to_dict())
