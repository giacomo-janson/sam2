"""
Module with mostly code for quickly using the SAM model at inference time.
"""
from pathlib import Path
import os
import time
import pathlib
import shutil
import subprocess
import numpy as np
import torch
from sam.utils import read_cfg_file, print_msg
from sam.nn.common import get_device
from sam.nn.autoencoder.decoder import get_decoder
from sam.nn.autoencoder.encoder import get_encoder
from sam.nn.noise_prediction.eps import get_eps_network
from sam.nn.generator import get_generative_model
from sam.data.cg_protein import EvalEncodedProteinDataset, MinimalDataset
from sam.data.topology import slice_ca_traj
try:
    from sam.data.aa_topology import get_traj_list
    from sam.data.aa_protein import AllAtomMinimalDataset
    import mdtraj
except ImportError:
    get_traj_list = None
    AllAtomMinimalDataset = None
from sam.data.topology import get_atom14_sam_data


class SAM:
    """
    Wrapper class for using the SAM models at inference time.
    """

    def __init__(
        self,
        config_fp: str,
        weights_dir: Path,
        device: str = "cpu",
        load_encoder: bool = False,
        verbose: bool = False
    ):

        _weights_dir = weights_dir.expanduser().resolve()
        

        self._check_sam()

        # self.weights_parent_path = weights_parent_path
        self.verbose = verbose

        self._print(f"# Setting up a SAM model from: {config_fp}.")

        #
        # Initial configuration.
        #

        # Read the configuration file of the model.
        self.model_cfg = read_cfg_file(config_fp)
        data_cfg = self.model_cfg.get("data", {})

        # PyTorch device.
        self.device, map_location_args = get_device(device, load=True)

        self._print("- Using device '%s'." % self.device)

        # Template-based model options.
        if "tbm" in data_cfg:
            if data_cfg["tbm"].get("mode") is None:
                self.use_tbm = False
            elif data_cfg["tbm"]["mode"] in ("single", ):
                self.use_tbm = True
            elif data_cfg["tbm"]["mode"] in ("random", ):
                self.use_tbm = True
                data_cfg["tbm"]["mode"] = "single"
            else:
                raise NotImplementedError(
                    "TBM mode {} not implemented for quick sampling.".format(
                        data_cfg["tbm"]["mode"]
                    ))
        else:
            self.use_tbm = False

        if self.use_tbm:
            self.tbm_type = data_cfg["tbm"].get("type", "enc")
        else:
            self.tbm_type = None

        # Protein large language model embeddings.
        self.use_pllm = data_cfg.get("pllm_embeddings_dp") is not None

        #
        # Load the epsilon network and the diffusion process object.
        #

        # Network.
        _gen_weights_fp = os.path.join(
            _weights_dir,
            "nn.gen.pt"
        )
        eps_fp = _gen_weights_fp  # self._get_weights_path(_gen_weights_fp)
        self._print(f"- Loading epsilon network from: {eps_fp}.")
        self.eps_model = get_eps_network(self.model_cfg)
        self.eps_model.load_state_dict(torch.load(eps_fp, **map_location_args))
        self.eps_model.to(self.device)
        if "ema" in self.model_cfg["generator"]:
            raise NotImplementedError()
        else:
            self.eps_ema = None

        # Load the standard scaler for the encodings, if necessary.
        _enc_scaler_fp = os.path.join(
            _weights_dir, "enc_std_scaler.pt"
        )
        if self.model_cfg["generative_stack"]["use_enc_scaler"]:
            enc_scaler_fp = _enc_scaler_fp  # self._get_weights_path(_enc_scaler_fp)
            enc_std_scaler = torch.load(enc_scaler_fp)
            enc_std_scaler["u"] = enc_std_scaler["u"].to(dtype=torch.float,
                                                         device=self.device)
            enc_std_scaler["s"] = enc_std_scaler["s"].to(dtype=torch.float,
                                                         device=self.device)
        else:
            enc_std_scaler = None
        self.enc_std_scaler = enc_std_scaler

        # Diffusion process.
        self.diffusion = get_generative_model(model_cfg=self.model_cfg,
                                              network=self.eps_model,
                                              ema=self.eps_ema)

        #
        # Load the autoencoder networks.
        #

        # Always load the decoder.
        _dec_weights_fp = os.path.join(
            _weights_dir, "nn.dec.pt"
        )
        dec_fp = _dec_weights_fp  # self._get_weights_path(_dec_weights_fp)
        self._print(f"- Loading decoder network from: {dec_fp}.")
        self.decoder = get_decoder(self.model_cfg)
        self.decoder.load_state_dict(torch.load(dec_fp, **map_location_args))
        self.decoder.to(self.device)

        # Load the encoder when using a template-based model.
        if self.tbm_type == "enc" or load_encoder:
            if "weights" in self.model_cfg["encoder"]:
                _enc_weights_fp = self.model_cfg["encoder"]["weights"]
            else:
                _enc_weights_fp = os.path.join(
                    data_cfg["exp_dp"], "ae", "nn.enc.pt")
            enc_fp = _enc_weights_fp  # self._get_weights_path(_enc_weights_fp)
            self._print(f"- Loading encoder network from: {enc_fp}.")
            self.encoder = get_encoder(self.model_cfg)
            self.encoder.load_state_dict(torch.load(enc_fp, **map_location_args))
            self.encoder.to(self.device)
        else:
            self.encoder = None
    
    def _check_sam(self):
        pass
    
    def _print(self, msg):
        print_msg(msg, verbose=self.verbose, tag="sampling")

    # def _get_weights_path(self, path):
    #     # 'weights_parent_path' takes precedence.
    #     if self.weights_parent_path is not None:
    #         return self._get_path(self.weights_parent_path, path)
    #     else:
    #         # If 'weights_parent_path' was not supplied, try to use a 'base_dp'
    #         # from the configuration file.
    #         if "data" in self.model_cfg and "base_dp" in self.model_cfg["data"]:
    #             return self._get_path(self.model_cfg["data"]["base_dp"], path)
    #         else:
    #             return path
    
    # def _get_path(self, parent, child):
    #     p = pathlib.Path(parent)
    #     c = pathlib.Path(child)
    #     return str(p / c)


    def sample(self,
        seq: str,
        n_samples: int = 1000,
        n_steps: int = 100,
        batch_size_eps: int = 256,
        batch_size_dec: int = None,
        tbm_data: dict = {},
        conditions: dict = {},
        aa_emb: torch.Tensor = None,
        prot_name: str = "protein",
        return_enc: bool = False,
        out_type: str = "numpy",
        sample_args: dict = {},
        use_cache: bool = False
        ):
        """
        Generates a Ca ensemble for a protein chain of sequence `seq`.

        Arguments
        `seq`: amino acid sequence of length L.
        `n_samples`: number of conformations to generate.
        `n_steps`: number of diffusion steps.
        `batch_size_eps`: batch size for sampling with the diffusion model.
        `batch_size_dec`: batch size for the decoding process.
        `tbm_data`: data used for template-based generative models. A dictionary
            which should contain a `xyz` key storing a (1, L, 3) array
            representing the coordinates of a template 3D structure.
        `conditions`: TODO.
        `prot_name`: name of the input protein sequence.
        `return_enc`: if 'True', returns also the generated encoding.
        `sample_args`: additional sampling arguments for the `sample` method
            of the generative model class used in the experiment. 

        Returns
        `out`: a dictionary storing the xyz coordinates in a tensor of shape
            (n_samples, len(seq), 3). If `return_enc` is 'True', also returns
            the generated encodings in a tensor of shape
            (n_samples, len(seq), enc_dim).
        """

        # Generate encodings.
        gen_out = self.generate(
            seq=seq,
            n_samples=n_samples,
            n_steps=n_steps,
            batch_size=batch_size_eps,
            tbm_data=tbm_data,
            conditions=conditions,
            aa_emb=aa_emb,
            prot_name=prot_name,
            sample_args=sample_args,
            use_cache=use_cache
        )

        # Decode to xyz coordinates.
        dec_out = self.decode(
            enc=gen_out["enc"],
            seq=seq,
            batch_size=batch_size_dec if batch_size_dec is not None \
                       else batch_size_eps,
            prot_name=prot_name,
            enc_traj=None if not sample_args.get("get_traj") \
                     else gen_out["enc_traj"]
        )

        # Return the output.
        out = {"seq": seq,
               "name": prot_name,
               "xyz": self._to(dec_out["xyz"], out_type),
               "time": {"tot": gen_out["time"]+dec_out["time"],
                        "ddpm": gen_out["time"],
                        "dec": dec_out["time"]}}
        if return_enc:
            out["enc"] = self._to(gen_out["enc"], out_type)

        if sample_args.get("get_traj"):
            if return_enc:
                out["enc_traj"] = self._to(gen_out["enc_traj"], out_type)
            out["xyz_traj"] = self._to(dec_out["xyz_traj"], out_type)
        
        return out


    def _to(self, t, out_type):
        if out_type == "numpy":
            return t.cpu().numpy()
        elif out_type == "torch":
            return t.cpu()
        else:
            raise TypeError(type(t))


    def generate(self,
            seq: str,
            n_samples: int = 1000,
            n_steps: int = 100,
            batch_size: int = 256,
            tbm_data: dict = {},
            conditions: dict = {},
            aa_emb: torch.Tensor = None,
            prot_name: str = "protein",
            sample_args: dict = {},
            use_cache: bool = False
        ):
        """
        Generate encodings for a protein with amino acid sequence 'seq'.
        """

        self._print(f"# Generating encodings for {n_samples} samples.")
        self._print(f"- seq: {seq}")

        gen_out = {}  # Dictionary to store the output of this method.

        # Encodings for templates.
        if self.use_tbm:
            
            # Encode a template xyz structure.
            if self.tbm_type == "enc":

                self._print("- Encoding the template structure.")
                
                if not "enc" in tbm_data:
                    # Dataset for decoding.
                    tem_dataset = self._get_tem_dataset(
                        name=prot_name,
                        seq=seq,
                        tbm_data=tbm_data,
                    )

                    # Dataloader for decoding.
                    tem_dataloader = torch.utils.data.dataloader.DataLoader(
                        dataset=tem_dataset, batch_size=1)

                    # Actually encode the template structure.
                    for batch in tem_dataloader:
                        batch = batch.to(self.device)
                        with torch.no_grad():
                            tem_enc = self.encoder.nn_forward(batch)

                # Use some existing encodings.
                else:
                    tem_enc = tbm_data["enc"]

                if self.enc_std_scaler is not None:
                    tem_enc = (tem_enc-self.enc_std_scaler["u"])/self.enc_std_scaler["s"]
                tem_enc = tem_enc.to("cpu")
            
            # Setup a template 3d structure.
            else:
                tem_traj = mdtraj.Trajectory(
                    xyz=tbm_data["xyz"],
                    topology=tbm_data["topology"]
                )
                tem_enc = get_atom14_sam_data(tem_traj)
            tbm = self.model_cfg["data"]["tbm"]

        else:
            # Don't use templates.
            tem_enc = None
            tbm = {"mode": None}

        # Dataset for the generated encodings.
        self._print("- Setting up a dataloader for encodings.")
        
        enc_dataset = EvalEncodedProteinDataset(
            name=prot_name,
            seq=seq,
            n_frames=n_samples,
            enc_dim=self.model_cfg["generative_stack"]["encoding_dim"],
            aa_emb=aa_emb,
            tbm=tbm,
            attributes=conditions,
            tem_enc=tem_enc,
            verbose=False
        )

        # Generated encodings dataloader.
        enc_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=enc_dataset, batch_size=batch_size
        )

        # Actually generate an encoded ensemble.
        self._print(f"- Generating...")
        encoding_dim = self.model_cfg["generative_stack"]["encoding_dim"]
        if not sample_args:
            sample_args = {"n_steps": n_steps}
        else:
            if "n_steps" in sample_args:
                raise ValueError()
            sample_args.update({"n_steps": n_steps})

        tot_graphs = 0
        time_gen = 0
        enc_gen = []
        enc_gen_traj = []  ##

        # EXTERNAL_POTENTIAL ###################################################
        if sample_args.get("bias"):  ##
            if self.model_cfg["generative_model"]["type"] != "diffusers_dm":  ##
                raise NotImplementedError()  ##
            self.diffusion.decoder = self.decoder  ##
            self.diffusion.enc_std_scaler = self.enc_std_scaler  ##
            use_bias = True  ##
        else:  ##
            use_bias = False  ##
        ########################################################################

        while tot_graphs < n_samples:
            for batch in enc_dataloader:
                batch = batch.to(self.device)
                time_gen_i = time.time()
                # EXTERNAL_POTENTIAL ###########################################
                if use_bias or sample_args.get("get_traj"):  ##
                    bias_out_i = self.diffusion.sample_bias(  ##
                        batch, use_cache=use_cache, **sample_args  ##
                    )  ##
                    # If using the diffusion trajectory.
                    if sample_args.get("get_traj"):  ##
                        enc_gen_i = bias_out_i["enc"]  ##
                        enc_gen_traj.append(bias_out_i["traj"])  ##
                    # Not using the diffusion trajectory.
                    else:  ##
                        enc_gen_i = bias_out_i  ##
                else:  ##
                    enc_gen_i = self.diffusion.sample(
                        batch, use_cache=use_cache, **sample_args
                    )
                ################################################################
                tot_time_i = time.time() - time_gen_i
                time_gen += tot_time_i
                # enc_gen_i = enc_gen_i.reshape(-1, n_nodes, encoding_dim)
                enc_gen.append(enc_gen_i)
                tot_graphs += batch.num_graphs
                self._print(
                    "- Generated %s conformations of %s in %s s" % (
                        tot_graphs, n_samples, tot_time_i
                    )
                )
                if tot_graphs >= n_samples:
                    break

        self._print(f"- Done.")

        # Prepare the output encodings.
        enc_gen = torch.cat(enc_gen, axis=0)[:n_samples]
        
        if self.enc_std_scaler is not None:  # Standardize if using a scaler.
            enc_gen = enc_gen*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]
        
        gen_out["enc"] = enc_gen
        gen_out["time"] = time_gen
        # EXTERNAL_POTENTIAL ###################################################
        if enc_gen_traj:  ##
            enc_gen_traj = torch.cat(enc_gen_traj, axis=1)  ##
            if self.enc_std_scaler is not None:  # Standardize if using a scaler.  ##
                enc_gen_traj = enc_gen_traj*self.enc_std_scaler["s"].unsqueeze(0) + self.enc_std_scaler["u"].unsqueeze(0)  ##
        # EXTERNAL_POTENTIAL ###################################################
        gen_out["enc_traj"] = enc_gen_traj

        return gen_out

    def _get_tem_dataset(self, name, seq, tbm_data):
        tem_dataset = MinimalDataset(
            name=name,
            seq=seq,
            xyz=tbm_data["xyz"],
            verbose=False
        )
        return tem_dataset


    def decode(self,
        enc: torch.Tensor,
        seq: str,
        batch_size: int = 256,
        prot_name: str = "protein",
        enc_traj: torch.Tensor = None
        ):
        """
        Decode generated encodings into xyz coordinates.
        """

        n_samples = enc.shape[0]

        self._print(f"# Decoding {n_samples} samples.")
        self._print("- Setting up a dataloader for xyz conformations.")

        dec_out = {}  # Dictionary to store the output of this method.

        # Dataset for decoding.
        dataset = MinimalDataset(
            name=prot_name,
            seq=seq,
            n_frames=n_samples,
            verbose=False
        )

        # Dataloader for decoding.
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset, batch_size=batch_size)

        # Actually decode the ensemble.
        time_gen = 0
        tot_graphs = 0
        xyz_gen = []
        xyz_traj = []
        self._print("- Decoding.")
        while tot_graphs < n_samples:
            for batch in dataloader:
                batch = batch.to(self.device)
                batch_y = torch.zeros(batch.x.shape[0],
                                      batch.x.shape[1],
                                      enc.shape[-1],
                                      device=self.device)
                e_gen_i = enc[tot_graphs:tot_graphs+batch.num_graphs]
                n_gen_i = e_gen_i.shape[0]
                pad_gen_batch = n_gen_i <= batch.num_graphs
                if pad_gen_batch:
                    batch_y[:e_gen_i.shape[0]] = e_gen_i
                else:
                    raise NotImplementedError()
                with torch.no_grad():
                    time_gen_i = time.time()
                    xyz_gen_i = self.decoder.nn_forward(batch_y, batch)
                    # EXTERNAL_POTENTIAL #######################################
                    if enc_traj is not None:
                        # xyz_gen_traj_i = []
                        # for t in range(enc_traj.shape[0]):
                        #     xyz_gen_traj_t = self.decoder.nn_forward(
                        #         enc_traj[t], batch
                        #     )
                        #     xyz_gen_traj_i.append(xyz_gen_traj_t.unsqueeze(0))
                        # xyz_traj.append(torch.cat(xyz_gen_traj_i, axis=0))
                        raise NotImplementedError()
                    ############################################################
                    time_gen += time.time() - time_gen_i
                if pad_gen_batch:
                    xyz_gen_i = xyz_gen_i[:n_gen_i]
                xyz_gen.append(xyz_gen_i)
                tot_graphs += xyz_gen_i.shape[0]
                self._print("- Decoded {} graphs of {}.".format(
                    tot_graphs, n_samples))
                if tot_graphs >= n_samples:
                    break
        
        self._print(f"- Done.")

        # Prepare the output xyz coordinates.
        xyz_gen = torch.cat(xyz_gen, axis=0)[:n_samples]

        dec_out["xyz"] = xyz_gen
        dec_out["time"] = time_gen
        # EXTERNAL_POTENTIAL ###################################################
        if xyz_traj:
            # xyz_traj = torch.cat(xyz_traj, axis=1)
            raise NotImplementedError()
        ########################################################################
        dec_out["xyz_traj"] = xyz_traj

        return dec_out
    

    def save(self,
        out: dict,
        out_path: str,
        out_fmt: str = "dcd"):

        self._print("# Saving output.")
        out_path = pathlib.Path(out_path)
        
        save_paths = {}

        # Save a FASTA file with the input sequence.
        fasta_path = out_path.parent / (out_path.name + ".seq.fasta")
        save_paths["fasta"] = fasta_path
        self._print(f"- Saving a FASTA sequence file to: {fasta_path}.")
        with open(fasta_path, "w") as o_fh:
            o_fh.write(f">{out['name']}\n{out['seq']}\n")

        # Save encodings.
        # enc_gen_path = out_path.parent / (out_path.name + ".enc.gen.npy")
        # np.save(enc_gen_path, out["enc"])

        # Save xyz coordinates.
        if out_fmt == "numpy":
            npy_path = out_path.parent / (out_path.name + ".ca.xyz.npy")
            save_paths["ca_npy"] = npy_path
            self._print(
                f"- Saving a C-alpha positions npy file to: {npy_path}.")
            np.save(npy_path, out["xyz"])
            
        elif out_fmt == "dcd":
            import mdtraj
            from sam.data.topology import get_ca_topology
            # Get the mdtraj C-alpha topology.
            topology = get_ca_topology(out["seq"])
            # Build a mdtraj C-alpha Trajectory.
            traj = mdtraj.Trajectory(xyz=out["xyz"], topology=topology)
            traj_path = out_path.parent / (out_path.name + ".ca.traj.dcd")
            # Save.
            save_paths["ca_dcd"] = traj_path
            pdb_path = out_path.parent / (out_path.name + ".ca.top.pdb")
            save_paths["ca_pdb"] = pdb_path
            self._print(
                f"- Saving a C-alpha trajectory dcd file to: {traj_path}.")
            self._print(
                f"- Saving a C-alpha topology PDB file to: {pdb_path}.")
            traj.save(str(traj_path))
            traj[0].save(str(pdb_path))

        else:
            raise KeyError(out_fmt)
        # EXTERNAL_POTENTIAL ###################################################
        if "enc_traj" in out:
            # np.save(
            #     out_path.parent / (out_path.name + ".enc.diffusion.npy"),
            #     out["enc_traj"],
            # )
            raise NotImplementedError()
        if "xyz_traj" in out:
            # np.save(
            #     out_path.parent / (out_path.name + ".ca.xyz.diffusion.npy"),
            #     out["xyz_traj"],
            # )
            raise NotImplementedError()
        ########################################################################
        return save_paths
    

class AllAtomSAM(SAM):
    """
    Class for generating all-atom ensembles.
    """

    def _check_sam(self):
        if get_traj_list is None:
            raise ImportError(
                "OpenFold is not installed. Cannot generate all-atom ensembles"
            )
    
    def _get_tem_dataset(self, name, seq, tbm_data):
        if AllAtomMinimalDataset is None:
            raise ImportError(
                "OpenFold is not installed. Cannot generate all-atom ensembles"
            )
        tem_dataset = AllAtomMinimalDataset(
            name=name,
            seq=seq,
            topology=tbm_data["topology"],
            xyz=tbm_data["xyz"],
            verbose=False
        )
        return tem_dataset

    def decode(self,
        enc: torch.Tensor,
        seq: str,
        batch_size: int = 256,
        prot_name: str = "protein",
        return_time: bool = False,
        enc_traj: torch.Tensor = None):
        """
        Decode generated encodings into xyz coordinates.
        """

        n_samples = enc.shape[0]

        self._print(f"# Decoding {n_samples} samples.")
        self._print("- Setting up a dataloader for xyz conformations.")

        # Dataset for decoding.
        dataset = MinimalDataset(
            name=prot_name,
            seq=seq,
            n_frames=n_samples,
            verbose=False
        )

        # Dataloader for decoding.
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset, batch_size=batch_size)

        # Actually decode the ensemble.
        time_gen = 0
        tot_graphs = 0
        xyz_gen = []
        xyz_traj = []
        xyz_traj_cls = []
        self._print("- Decoding.")
        while tot_graphs < n_samples:
            for batch in dataloader:
                batch = batch.to(self.device)
                batch_y = torch.zeros(batch.x.shape[0],
                                      batch.x.shape[1],
                                      enc.shape[-1],
                                      device=self.device)
                e_gen_i = enc[tot_graphs:tot_graphs+batch.num_graphs]
                n_gen_i = e_gen_i.shape[0]
                batch_y[:e_gen_i.shape[0]] = e_gen_i
                with torch.no_grad():
                    time_gen_i = time.time()
                    sm_i = self.decoder.nn_forward(batch_y, batch)
                    time_gen += time.time() - time_gen_i
                # EXTERNAL_POTENTIAL #######################################
                if enc_traj is not None:
                    xyz_gen_traj_i = []
                    xyz_traj_cls_i = []
                    for t in range(enc_traj.shape[0]):
                        with torch.no_grad():
                            enc_gen_it = enc_traj[t][tot_graphs:tot_graphs+batch.num_graphs]
                            xyz_gen_traj_t = self.decoder.nn_forward(
                                enc_gen_it, batch
                            )
                        xyz_gen_traj_t["positions"] = xyz_gen_traj_t["positions"][-1]
                        xyz_gen_traj_i.append(xyz_gen_traj_t)
                        xyz_traj_cls_i.extend(
                            [m for m in range(0+tot_graphs, batch.num_graphs+tot_graphs)]
                        )
                    xyz_traj.extend(xyz_gen_traj_i)
                    xyz_traj_cls.extend(xyz_traj_cls_i)
                ############################################################
                # if pad_gen_batch:
                #     xyz_gen_i = xyz_gen_i[:n_gen_i]
                for k in sm_i.keys():
                    sm_i[k] = sm_i[k].cpu()
                sm_i["positions"] = sm_i["positions"][-1]
                xyz_gen.append(sm_i)
                tot_graphs += n_gen_i
                self._print("- Decoded {} graphs of {}.".format(
                    tot_graphs, n_samples))
                if tot_graphs >= n_samples:
                    break
        
        self._print(f"- Done.")

        traj_gen = []
        for sm_i in xyz_gen:
            traj_i = get_traj_list(sm_i)
            traj_gen.extend(traj_i)
        traj_gen = mdtraj.join(traj_gen)
        traj_gen = traj_gen[:n_samples]

        results = {"xyz": traj_gen, "time": time_gen}
        # EXTERNAL_POTENTIAL ###################################################
        if xyz_traj:
            traj_gen_diffusion = []
            xyz_traj_cls = np.array(xyz_traj_cls)
            print(xyz_traj_cls, len(xyz_traj_cls))
            print(len(xyz_traj))
            for sm_i in xyz_traj:
                traj_gen_diffusion.extend(get_traj_list(sm_i))
            traj_gen_diffusion = mdtraj.join(traj_gen_diffusion)
            results["xyz_traj"] = []
            for m in range(tot_graphs):
                results["xyz_traj"].append(traj_gen_diffusion[xyz_traj_cls == m])
        ########################################################################

        return results
    
    def _to(self, t, out_type):
        return t

    def save(self,
            out: dict,
            out_path: str,
            out_fmt: str = "dcd",
            save_ca: bool = False
        ):

        self._print("# Saving output.")
        out_path = pathlib.Path(out_path)
        
        save_paths = {}

        # Save a FASTA file with the input sequence.
        fasta_path = out_path.parent / (out_path.name + ".seq.fasta")
        save_paths["fasta"] = fasta_path
        # self._print(f"- Saving a FASTA sequence file to: {fasta_path}.")
        # with open(fasta_path, "w") as o_fh:
        #     o_fh.write(f">{out['name']}\n{out['seq']}\n")

        # Save encodings.
        # enc_gen_path = out_path.parent / (out_path.name + ".enc.gen.npy")
        # np.save(enc_gen_path, out["enc"])

        # Save xyz coordinates.
        if out_fmt == "numpy":
            raise NotImplementedError()
            
        elif out_fmt in ("dcd", "xtc"):
            parent_path = out_path.parent
            traj_path = parent_path / (out_path.name + f".traj.{out_fmt}")
            pdb_path = parent_path / (out_path.name + ".top.pdb")
            self._print(f"- Saving a trajectory file to: {traj_path}.")
            out["xyz"].save(str(traj_path))
            self._print(f"- Saving a topology PDB file to: {pdb_path}.")
            out["xyz"][0].save(str(pdb_path))
            save_paths["aa_traj"] = str(traj_path)
            save_paths["aa_top"] = str(pdb_path)
            if save_ca:
                ca_traj_path = parent_path / (out_path.name + f".ca.traj.{out_fmt}")
                ca_pdb_path = parent_path / (out_path.name + ".ca.top.pdb")
                ca_traj = slice_ca_traj(out["xyz"])
                self._print(f"- Saving a Ca trajectory file to: {ca_traj_path}.")
                ca_traj.save(str(ca_traj_path))
                self._print(f"- Saving a Ca topology PDB file to: {ca_pdb_path}.")
                ca_traj[0].save(str(ca_pdb_path))
                save_paths["ca_traj"] = str(ca_traj_path)
                save_paths["ca_top"] = str(ca_pdb_path)

            # EXTERNAL_POTENTIAL ###############################################
            if "xyz_traj" in out:
                for m, traj_diffusion in enumerate(out["xyz_traj"]):
                    traj_path = parent_path / (out_path.name + f".aa.diffusion.{m}.{out_fmt}")
                    traj_diffusion.save(str(traj_path))
            ####################################################################

        else:
            raise KeyError(out_fmt)
        
        return save_paths
