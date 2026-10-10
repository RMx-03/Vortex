import importlib.util
import json
import sys
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('publication',P/'scripts/publication.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
pol=m.policy()

class PublicationTests(unittest.TestCase):
    def test_safe_public_file(self):
        self.assertIsNone(m.safe_blob('docs/architecture.md','100644',b'# Architecture\n',pol))
    def test_rejects_private_path(self):
        self.assertIsNotNone(m.safe_blob('docs/sessions/notes.md','100644',b'x',pol))
        self.assertIsNotNone(m.safe_blob('AGENTS.md','100644',b'x',pol))
    def test_rejects_symlink(self):
        self.assertIsNotNone(m.safe_blob('docs/safe.md','120000',b'../../secret',pol))
    def test_rejects_credentials_without_echo(self):
        reason=m.safe_blob('docs/a.md','100644',b'AKIA' + b'Q'*16,pol)
        self.assertIn('redacted',reason)
        self.assertNotIn('AKIA',reason)
    def test_rejects_notebook_outputs(self):
        n=clean_notebook()
        n['cells'][0]['outputs']=[{'output_type':'stream','name':'stdout','text':'oops'}]
        self.assertIn('outputs',m.safe_blob(NOTEBOOK,'100644',json.dumps(n).encode(),pol))
    def test_safe_clean_notebook(self):
        self.assertIsNone(m.safe_blob(NOTEBOOK,'100644',json.dumps(clean_notebook()).encode(),pol))
    def test_commit_and_ref(self):
        self.assertIsNone(m.check_text('commit','feat(scheduler): implement FIFO selection'))
        self.assertIsNotNone(m.check_text('commit','random message'))
        self.assertIsNotNone(m.check_text('commit','feat(core): change thing\nCo-authored-by: Unrequested'))
        self.assertIsNone(m.check_text('ref','refs/heads/feat/scheduler-fifo'))
        self.assertIsNotNone(m.check_text('ref','refs/heads/something bad'))
    def test_private_header(self):
        self.assertIsNotNone(m.safe_blob('docs/x.md','100644',b'Classification: PRIVATE - DO NOT PUBLISH',pol))
    def test_normal_technical_terms_allowed(self):
        self.assertIsNone(m.safe_blob('docs/architecture.md','100644',b'An autonomous agent sends a model request.',pol))

NOTEBOOK='notebooks/colab/00_gpu_check.ipynb'


def clean_notebook():
    return {'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[
        {'id':'synthetic-code','cell_type':'code','metadata':{},'source':['print("CPU probe")\n'],
         'outputs':[],'execution_count':None}]}


class PublicationRegressionTests(unittest.TestCase):
    def test_environment_variants_and_hidden_local_directories(self):
        for path in ['src/vortex/.venv311/secret.py','tests/unit/.venv-gpu/pkg.py',
                     'docs/.hidden/note.md','src/vortex/venv311/pkg.py']:
            with self.subTest(path=path):
                self.assertFalse(m.safe_path(path,pol))

    def test_windows_and_noncanonical_path_escapes(self):
        for path in ['docs//a.md','docs/../a.md','docs/./a.md','docs/context./a.md',
                     'docs/context /a.md','docs/a:stream.md','docs/CON.md',
                     'docs/aux/a.md','docs/a\tb.md','docs/a\u202eb.md',
                     '/docs/a.md','docs\\a.md','C:/docs/a.md']:
            with self.subTest(path=path):
                self.assertFalse(m.safe_path(path,pol))

    def test_secret_shaped_filename_is_blocked_without_echo(self):
        token='ghp_'+'Q'*30
        issue=m.safe_blob('docs/'+token+'.md','100644',b'ordinary text',pol)
        self.assertIsNotNone(issue)
        self.assertNotIn(token,issue)

    def test_markdown_private_classification_is_detected(self):
        for content in ['# Private — do not publish','**Classification:** PRIVATE',
                        '# Notes\n\nClassification: PRIVATE - DO NOT PUBLISH']:
            with self.subTest(content=content):
                self.assertIsNotNone(m.safe_blob('docs/a.md','100644',content.encode(),pol))

    def test_extended_credentials_are_redacted(self):
        values=['ASIA'+'Q'*16,'sk-proj-'+'q'*40,'sk-ant-api03-'+'q'*40]
        for value in values:
            with self.subTest(prefix=value[:4]):
                issue=m.safe_blob('docs/a.md','100644',value.encode(),pol)
                self.assertIsNotNone(issue)
                self.assertNotIn(value,issue)

    def test_binary_invalid_utf8_size_and_submodule_are_blocked(self):
        for mode,data in [('160000',b''),('100644',b'\x00'),
                          ('100644',b'\xff'),('100644',b'x'*(pol['maximum_text_bytes']+1))]:
            with self.subTest(mode=mode,size=len(data)):
                self.assertIsNotNone(m.safe_blob('docs/a.md',mode,data,pol))

    def test_notebook_schema_and_metadata_outputs_are_rejected(self):
        invalid=[
            {'nbformat':4,'cells':[]},
            {**clean_notebook(),'cells':None},
            {**clean_notebook(),'cells':[None]},
            {**clean_notebook(),'cells':[{'cell_type':'unknown','metadata':{},'source':[]}]},
        ]
        for field,value in [('outputs',{}),('execution_count',0),
                            ('source',[1]),('metadata',[]),('attachments',{'x':{}})]:
            obj=clean_notebook()
            obj['cells'][0][field]=value
            invalid.append(obj)
        obj=clean_notebook()
        obj['cells'][0]['metadata']['widgets']={'state':'synthetic saved output'}
        invalid.append(obj)
        for obj in invalid:
            with self.subTest(obj=obj):
                self.assertIsNotNone(m.safe_blob(NOTEBOOK,'100644',json.dumps(obj).encode(),pol))

    def test_duplicate_notebook_keys_and_nonfinite_values_are_rejected(self):
        for data in [b'{"nbformat":4,"nbformat":3,"cells":[]}',
                     json.dumps(clean_notebook()).replace('{}','{"value":NaN}',1).encode()]:
            self.assertIsNotNone(m.safe_blob(NOTEBOOK,'100644',data,pol))

    def test_legitimate_technical_docs_and_markdown_notebook_are_allowed(self):
        text=b'AI agents send model requests. Generated token IDs are measured. Codex is mentioned as a tool.'
        self.assertIsNone(m.safe_blob('docs/a.md','100644',text,pol))
        obj=clean_notebook()
        obj['cells'].append({'id':'synthetic-markdown','cell_type':'markdown','metadata':{},'source':'# GPU probe'})
        self.assertIsNone(m.safe_blob(NOTEBOOK,'100644',json.dumps(obj).encode(),pol))

    def test_markdown_attribution_lines_are_blocked(self):
        for value in ['**Generated-by:** Synthetic','Generated by Codex',
                      'Created with Claude Code','Implemented by Google Antigravity']:
            self.assertIsNotNone(m.check_text('prose',value))

    def test_notebook_decoded_metadata_cannot_hide_secrets_or_private_labels(self):
        for value in ['ghp_'+'Q'*30,'Classification: PRIVATE','Generated by Codex']:
            obj=clean_notebook()
            obj['metadata']['description']=value
            # Every character is escaped, so raw JSON regex matching is insufficient.
            data=json.dumps(obj).replace(json.dumps(value),
                '"'+''.join('\\u%04x'%ord(c) for c in value)+'"').encode()
            issue=m.safe_blob(NOTEBOOK,'100644',data,pol)
            self.assertIsNotNone(issue)
            self.assertNotIn(value,issue)

    def test_notebook_cell_ids_and_unknown_payload_fields_are_validated(self):
        invalid=[]
        obj=clean_notebook()
        del obj['cells'][0]['id']
        invalid.append(obj)
        for value in ['',[], 'a'*65,'space invalid']:
            obj=clean_notebook()
            obj['cells'][0]['id']=value
            invalid.append(obj)
        obj=clean_notebook()
        obj['cells'].append(dict(obj['cells'][0]))
        invalid.append(obj)
        obj=clean_notebook()
        obj['cells'][0]['unexpected_payload']='synthetic saved output'
        invalid.append(obj)
        obj=clean_notebook()
        obj['unexpected_payload']='synthetic saved output'
        invalid.append(obj)
        for obj in invalid:
            with self.subTest(obj=obj):
                self.assertIsNotNone(m.safe_blob(NOTEBOOK,'100644',json.dumps(obj).encode(),pol))

    def test_public_json_decoded_strings_and_syntax_are_checked(self):
        for value in ['ghp_'+'Q'*30,'Classification: PRIVATE','Generated by Codex']:
            data=json.dumps({'note':value}).replace(json.dumps(value),
                '"'+''.join('\\u%04x'%ord(c) for c in value)+'"').encode()
            self.assertIsNotNone(m.safe_blob('benchmarks/workloads/example.json','100644',data,pol))
        self.assertIsNotNone(m.safe_blob('benchmarks/workloads/example.json','100644',b'{',pol))
        self.assertIsNone(m.safe_blob('benchmarks/workloads/example.json','100644',
                                    b'{"description":"AI agents send requests", "tokens":128}',pol))

    def test_known_notebook_metadata_fields_are_validated(self):
        invalid=[('root','kernelspec',[]),('root','kernelspec',{'name':123}),
                 ('root','language_info',{}),('root','title',[]),
                 ('cell','tags',[1]),('cell','tags',['repeat','repeat']),
                 ('cell','collapsed',[]),('cell','jupyter',[]),
                 ('cell','execution',{'time':1})]
        for location,key,value in invalid:
            obj=clean_notebook()
            metadata=obj['metadata'] if location=='root' else obj['cells'][0]['metadata']
            metadata[key]=value
            with self.subTest(location=location,key=key):
                self.assertIsNotNone(m.safe_blob(NOTEBOOK,'100644',json.dumps(obj).encode(),pol))
        obj=clean_notebook()
        obj['metadata'].update(kernelspec={'name':'python3','display_name':'Python 3'},
                               language_info={'name':'python'},custom={'project':'Vortex'})
        obj['cells'][0]['metadata'].update(tags=['probe'],collapsed=False,jupyter={'source_hidden':False})
        self.assertIsNone(m.safe_blob(NOTEBOOK,'100644',json.dumps(obj).encode(),pol))


class DisposableRepositoryTests(unittest.TestCase):
    """Real Git fixtures never inherit the user's hooks, identity or index."""

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='vortex publication test ')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'repo with spaces'
        self.root.mkdir()
        self.env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        self.env.update(GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_NOSYSTEM='1',
                        PYTHONDONTWRITEBYTECODE='1')
        template=Path(self.temp.name)/'empty-template'
        template.mkdir()
        self.git('init','-b','main','--template='+str(template))
        self.git('config','user.name','Synthetic Fixture')
        self.git('config','user.email','fixture@example.invalid')
        self.git('config','core.hooksPath',str(self.root/'.git/disabled-hooks'))
        self.git('config','core.autocrlf','false')
        self.git('config','core.ignorecase','false')
        (self.root/'scripts').mkdir()
        (self.root/'.publication').mkdir()
        shutil.copy2(P/'scripts/publication.py',self.root/'scripts/publication.py')
        shutil.copy2(P/'.publication/policy.json',self.root/'.publication/policy.json')
        self.write('README.md',b'# Synthetic fixture\n')
        self.git('add','--','README.md','.publication/policy.json')
        self.git('commit','-m','chore(test): initialize fixture')
        self.base=self.git('rev-parse','HEAD').stdout.strip()

    def git(self,*args,input=None):
        run=subprocess.run(['git','-C',str(self.root),*args],input=input,
                           capture_output=True,env=self.env)
        if run.returncode:
            self.fail('Disposable Git fixture failed: '+args[0]+' (no user repository modified)')
        run.stdout=run.stdout.decode('utf-8')
        return run

    def write(self,path,data):
        target=self.root/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)

    def add(self,path,data):
        self.write(path,data)
        self.git('add','-f','--',path)

    def index_object(self,path,mode='100644',data=b'# synthetic\n'):
        oid=self.git('hash-object','-w','--stdin',input=data).stdout.strip()
        if mode=='160000':
            oid=self.base
        self.git('update-index','--add','--cacheinfo',mode+','+oid+','+path)
        return oid

    def scan(self,*args,input=None,env=None,script=None):
        return subprocess.run([sys.executable,str(script or self.root/'scripts/publication.py'),*args],
                              cwd=self.root,input=input,capture_output=True,text=True,
                              encoding='utf-8',env=env or self.env)

    def assert_scan(self,code,*args,input=None):
        run=self.scan(*args,input=input)
        self.assertEqual(run.returncode,code,run.stdout+run.stderr)
        self.assertNotIn('Traceback',run.stdout+run.stderr)
        return run

    def update_input(self,local=None,remote=None,local_ref='refs/heads/main',
                     remote_ref='refs/heads/main'):
        return '{} {} {} {}\n'.format(local_ref,local or self.git('rev-parse','HEAD').stdout.strip(),
                                     remote_ref,remote or '0'*len(self.base))

    def test_staged_private_blob_blocked_after_worktree_sanitized(self):
        self.add('docs/a.md',b'Classification: PRIVATE - DO NOT PUBLISH\nsynthetic\n')
        self.write('docs/a.md',b'# sanitized working file\n')
        before=(self.root/'.git/index').read_bytes()
        self.assert_scan(1,'staged')
        self.assertEqual((self.root/'.git/index').read_bytes(),before)

    def test_safe_staged_blob_accepted_despite_unsafe_worktree(self):
        self.add('docs/a.md',b'# safe staged fixture\n')
        self.write('docs/a.md',b'Classification: PRIVATE\nsynthetic\n')
        self.assert_scan(0,'staged')

    def test_staged_symlink_and_submodule_are_blocked_without_windows_symlink_privilege(self):
        self.index_object('docs/link.md','120000',b'../../outside')
        self.index_object('src/vortex/submodule.py','160000')
        run=self.assert_scan(1,'staged')
        self.assertIn('file mode',run.stdout)

    def test_case_collision_in_index_and_committed_tree(self):
        self.index_object('docs/Case.md')
        self.index_object('docs/case.md')
        self.assert_scan(1,'staged')
        self.git('commit','-m','docs(test): create synthetic collision')
        self.assert_scan(1,'tree','HEAD')

    def test_unmerged_index_is_blocked(self):
        oid=self.git('hash-object','-w','--stdin',input=b'# fixture\n').stdout.strip()
        self.git('update-index','--index-info',
                 input=('100644 '+oid+' 1\tdocs/merge.md\n').encode())
        self.assert_scan(1,'staged')

    def test_secret_in_filename_is_never_printed(self):
        secret='ghp_'+'Q'*30
        self.index_object('docs/'+secret+'.md')
        run=self.assert_scan(1,'staged')
        self.assertNotIn(secret,run.stdout+run.stderr)

    def test_private_ancestral_blob_blocked_even_after_deletion(self):
        self.add('docs/leak.md',b'Classification: PRIVATE\nSynthetic fixture only.\n')
        self.git('commit','-m','docs(test): add synthetic private fixture')
        self.git('rm','--','docs/leak.md')
        self.git('commit','-m','docs(test): delete fixture')
        self.assert_scan(0,'tree','HEAD')
        self.assert_scan(1,'outgoing',input=self.update_input(remote=self.base))

    def test_outgoing_checks_remote_destination_and_delete_ref(self):
        self.assert_scan(1,'outgoing',input=self.update_input(remote_ref='refs/heads/unapproved'))
        self.assert_scan(1,'outgoing',input=self.update_input(
            local='0'*len(self.base),remote=self.base,local_ref='(delete)',
            remote_ref='refs/heads/unapproved'))

    def test_secret_shaped_reference_is_redacted(self):
        secret='ghp_'+'Q'*30
        run=self.assert_scan(1,'outgoing',input=self.update_input(
            local_ref='refs/heads/chore/'+secret))
        self.assertNotIn(secret,run.stdout+run.stderr)

    def test_unsafe_annotated_tag_is_scanned(self):
        self.git('tag','-a','v1.0.0','-m','Classification: PRIVATE\nSynthetic tag fixture.')
        oid=self.git('rev-parse','refs/tags/v1.0.0').stdout.strip()
        self.assert_scan(1,'outgoing',input=self.update_input(
            local=oid,local_ref='refs/tags/v1.0.0',remote_ref='refs/tags/v1.0.0'))

    def test_safe_annotated_tag_and_lightweight_tag_are_allowed(self):
        self.git('tag','-a','v1.0.0','-m','Safe release fixture')
        oid=self.git('rev-parse','refs/tags/v1.0.0').stdout.strip()
        self.assert_scan(0,'outgoing',input=self.update_input(
            local=oid,local_ref='refs/tags/v1.0.0',remote_ref='refs/tags/v1.0.0'))
        self.assert_scan(0,'outgoing',input=self.update_input(
            local_ref='refs/tags/v1.0.1',remote_ref='refs/tags/v1.0.1'))

    def test_tag_to_noncommit_object_is_fail_closed(self):
        oid=self.git('hash-object','-w','--stdin',input=b'synthetic blob\n').stdout.strip()
        self.assert_scan(2,'outgoing',input=self.update_input(
            local=oid,local_ref='refs/tags/v1.0.0',remote_ref='refs/tags/v1.0.0'))

    def test_shallow_history_is_rejected(self):
        (self.root/'.git/shallow').write_text(self.base+'\n',encoding='ascii')
        self.assert_scan(2,'outgoing',input=self.update_input())

    def test_git_replace_cannot_hide_private_intermediate_commit(self):
        self.add('docs/leak.md',b'Classification: PRIVATE\nsynthetic\n')
        self.git('commit','-m','docs(test): add private fixture')
        leaky=self.git('rev-parse','HEAD').stdout.strip()
        self.git('rm','--','docs/leak.md')
        self.git('commit','-m','docs(test): remove private fixture')
        self.git('replace',leaky,self.base)
        self.assert_scan(1,'outgoing',input=self.update_input())

    def test_graft_history_is_rejected(self):
        graft=self.root/'.git/info/grafts'
        graft.parent.mkdir(exist_ok=True)
        graft.write_text(self.base+'\n',encoding='ascii')
        self.assert_scan(2,'outgoing',input=self.update_input())

    def test_missing_remote_history_and_malformed_updates_are_errors(self):
        self.assert_scan(2,'outgoing',input=self.update_input(remote='f'*len(self.base)))
        for value in ['missing fields\n','refs/heads/main --all refs/heads/main '+ '0'*40+'\n']:
            self.assert_scan(2,'outgoing',input=value)

    def test_policy_schema_errors_have_actionable_no_traceback_failure(self):
        path=self.root/'.publication/policy.json'
        for value in ['[]','null','{"schema_version":1,"allowed_exacts":["README.md"]}',
                      '{"schema_version":1,"schema_version":1}']:
            path.write_text(value,encoding='utf-8')
            run=self.assert_scan(2,'staged')
            self.assertIn('policy',run.stderr.lower())

    def test_nested_checkout_root_is_rejected(self):
        nested=self.root/'nested'
        (nested/'scripts').mkdir(parents=True)
        (nested/'.publication').mkdir()
        shutil.copy2(P/'scripts/publication.py',nested/'scripts/publication.py')
        shutil.copy2(P/'.publication/policy.json',nested/'.publication/policy.json')
        run=self.scan('staged',script=nested/'scripts/publication.py')
        self.assertEqual(run.returncode,2)
        self.assertNotIn('Traceback',run.stderr)

    def test_all_supported_cli_modes_accept_representative_safe_inputs(self):
        self.assert_scan(0,'staged')
        self.assert_scan(0,'tree','HEAD')
        self.assert_scan(0,'outgoing',input=self.update_input())
        self.assert_scan(0,'text','An AI agent sends inference requests.')
        self.assert_scan(0,'ref','refs/heads/chore/scaffold-foundation')
        msg=Path(self.temp.name)/'safe-message.txt'
        msg.write_text('chore(security): add publication safeguards\n',encoding='utf-8')
        self.assert_scan(0,'commit-msg',str(msg))

    def test_cli_blocks_secret_text_unsafe_ref_trailer_and_bad_notebook(self):
        secret='ghp_'+'Q'*30
        run=self.assert_scan(1,'text',secret)
        self.assertNotIn(secret,run.stdout+run.stderr)
        self.assert_scan(1,'ref','refs/heads/unapproved')
        msg=Path(self.temp.name)/'bad-message.txt'
        msg.write_text('docs(test): fixture\n'+'Co-'+'authored-by: Synthetic\n',encoding='utf-8')
        self.assert_scan(1,'commit-msg',str(msg))
        obj=clean_notebook()
        obj['cells'][0]['outputs']=[{'output_type':'stream','name':'stdout','text':'fixture'}]
        self.add(NOTEBOOK,json.dumps(obj).encode())
        self.assert_scan(1,'staged')

    def test_directory_case_collisions_are_rejected(self):
        self.index_object('docs/Example/a.md')
        self.index_object('docs/example/b.md')
        self.assert_scan(1,'staged')
        self.git('commit','-m','docs(test): synthetic directory collision')
        self.assert_scan(1,'tree','HEAD')

    def test_oversized_blob_and_history_limit_are_fail_closed(self):
        self.add('docs/large.md',b'x'*(pol['maximum_text_bytes']+1))
        self.assert_scan(1,'staged')
        self.git('reset','--','docs/large.md')
        self.add('docs/a.md',b'# safe\n')
        self.git('commit','-m','docs(test): safe second commit')
        changed=dict(pol,max_commits_per_push_scan=1)
        self.write('.publication/policy.json',json.dumps(changed).encode())
        self.assert_scan(2,'outgoing',input=self.update_input())

    def test_external_index_and_option_shaped_tree_are_rejected(self):
        env=dict(self.env,GIT_INDEX_FILE=str(Path(self.temp.name)/'outside-index'))
        run=self.scan('staged',env=env)
        self.assertEqual(run.returncode,2,run.stdout+run.stderr)
        self.assert_scan(2,'tree','--','--all')

    def test_outgoing_head_raw_oid_multi_ref_and_merge_are_supported(self):
        for ref in ['HEAD',self.base]:
            self.assert_scan(0,'outgoing',input=self.update_input(local_ref=ref))
        self.git('checkout','-b','docs/fixture')
        self.add('docs/a.md',b'# safe side branch\n')
        self.git('commit','-m','docs(test): safe side branch')
        self.git('checkout','main')
        self.add('docs/b.md',b'# safe main branch\n')
        self.git('commit','-m','docs(test): safe main branch')
        self.git('merge','--no-ff','docs/fixture','-m','chore(test): merge safe fixture')
        self.assert_scan(0,'outgoing',input=self.update_input()+self.update_input(
            remote_ref='refs/heads/docs/fixture',remote=self.base))

    def test_nested_annotated_tag_cannot_hide_private_annotation(self):
        self.git('tag','-a','v1.0.0','-m','Classification: PRIVATE\nSynthetic fixture.')
        self.git('tag','-a','v1.0.1','v1.0.0','-m','Safe outer fixture')
        oid=self.git('rev-parse','refs/tags/v1.0.1').stdout.strip()
        self.assert_scan(1,'outgoing',input=self.update_input(
            local=oid,local_ref='refs/tags/v1.0.1',remote_ref='refs/tags/v1.0.1'))

    def test_commit_author_credentials_are_redacted(self):
        secret='ghp_'+'Q'*30
        self.git('config','user.name',secret)
        self.add('docs/a.md',b'# safe document\n')
        self.git('commit','-m','docs(test): synthetic metadata leak')
        run=self.assert_scan(1,'outgoing',input=self.update_input(remote=self.base))
        self.assertNotIn(secret,run.stdout+run.stderr)

    def test_git_environment_cannot_redirect_checkout_to_clean_repository(self):
        self.add('docs/private.md',b'Classification: PRIVATE\nsynthetic\n')
        other=Path(self.temp.name)/'other'
        subprocess.run(['git','init','--bare',str(other)],env=self.env,
                       capture_output=True,check=True)
        env=dict(self.env,GIT_DIR=str(other),GIT_WORK_TREE=str(self.root))
        run=self.scan('staged',env=env)
        self.assertEqual(run.returncode,2,run.stdout+run.stderr)

    def test_malformed_annotated_tag_headers_are_errors(self):
        valid=['object '+self.base,'type commit','tag v1.0.0',
               'tagger Synthetic Fixture <fixture@example.invalid> 1700000000 +0000']
        variants=[valid[:1]+valid[2:],valid[:1]+['type blob']+valid[2:],valid[:3]]
        for lines in variants:
            raw=('\n'.join(lines)+'\n\nSafe annotation\n').encode()
            oid=self.git('hash-object','--literally','-w','-t','tag','--stdin',input=raw).stdout.strip()
            self.assert_scan(2,'outgoing',input=self.update_input(
                local=oid,local_ref='refs/tags/v1.0.0',remote_ref='refs/tags/v1.0.0'))

    def test_promisor_checkout_is_rejected_before_any_lazy_fetch(self):
        self.git('config','remote.fixture.url','https://example.invalid/never-contact')
        self.git('config','remote.fixture.promisor','true')
        for mode in ['staged','tree','outgoing']:
            run=self.assert_scan(2,mode,input=self.update_input() if mode=='outgoing' else None)
            self.assertIn('partial',run.stderr.lower())

    def test_linked_worktree_and_git_owned_temporary_index_are_supported(self):
        linked=Path(self.temp.name)/'linked worktree'
        self.git('worktree','add','--detach',str(linked),self.base)
        (linked/'scripts').mkdir()
        shutil.copy2(P/'scripts/publication.py',linked/'scripts/publication.py')
        run=self.scan('staged',script=linked/'scripts/publication.py')
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        temporary=self.root/'.git/temporary-index'
        shutil.copy2(self.root/'.git/index',temporary)
        env=dict(self.env,GIT_INDEX_FILE=str(temporary))
        run=self.scan('staged',env=env)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)


if __name__=='__main__':
    unittest.main()
