function FlagMake = mkdirIfNotExist(Dir)
	if ~exist(Dir, 'dir')
		mkdir(Dir); FlagMake = true;
	else
		FlagMake = false;
	end
end