    def _create_refine_restraints(self, multiconformer):
        """
        For refinement, we need to create occupancy restraints for all residues in the same segment with the same altloc.
        This function will go through the qFit output and create a constraint file to be fed into refinement
        """
        fname = os.path.join(self.options.directory, "qFit_occupancy.params")
        f = open(fname, "w+")
        f.write("refinement {\n")
        f.write("  refine {\n")
        f.write("    occupancies {\n")
        resi_ = []
        altloc_ = []
        chain_ = []
        for chain in multiconformer:
            for residue in chain:
                if residue.resn[0] in ROTAMERS:
                    if len(residue.extract("name", "CA", "==").q) == 1:
                        # if something exists in the list, print it
                        if (
                            len(resi_) > 0
                        ):  # something exists and we should write it out
                            for a in set(altloc_):
                                # create a string from each value in the resi array
                                # resi_str = ','.join(map(str, resi_))
                                f.write("      constrained_group {\n")
                                for l in range(0, len(resi_)):
                                    # make string for each residue and concat the strings together
                                    if l == 0:
                                        resi_selection = f"((chain {chain_[0]} and resseq {resi_[l]})"  # first residue
                                    else:
                                        resi_selection = (
                                            resi_selection
                                            + f" or (chain {chain_[0]} and resseq {resi_[l]})"
                                        )
                                f.write(
                                    f"        selection = altid {a} and {resi_selection})\n"
                                )
                                f.write("             }\n")
                        resi_ = []
                        altloc_ = []
                        chain_ = []
                    else:
                        for alt in list(set(residue.altloc)):
                            # only append if it does not already exist in array
                            if residue.resi[0] not in resi_:
                                resi_.append(residue.resi[0])
                            chain_.append(residue.chain[0])
                            altloc_.append(alt[0])
        f.write("   }\n")
        f.write(" }\n")
        f.write("}\n")
        f.close()